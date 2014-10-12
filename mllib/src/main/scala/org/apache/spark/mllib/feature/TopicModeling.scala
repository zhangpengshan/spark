/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import java.util.Random

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum}

import org.apache.spark.annotation.Experimental
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext._

import TopicModeling._

class TopicModeling private[mllib](
  @transient var corpus: Graph[VD, ED],
  val numTopics: Int,
  val numTerms: Int,
  val alpha: Double,
  val beta: Double,
  @transient val storageLevel: StorageLevel)
  extends Serializable with Logging {

  def this(docs: RDD[(TopicModeling.DocId, SSV)],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    computedModel: Broadcast[TopicModel] = null) {
    this(initializeCorpus(docs, numTopics, storageLevel, computedModel),
      numTopics, docs.first()._2.size, alpha, beta, storageLevel)
  }


  /**
   * The number of documents in the corpus
   */
  val numDocs = docVertices.count()

  /**
   * The number of terms in the corpus
   */
  private val sumTerms = corpus.edges.map(e => e.attr.size.toDouble).sum().toLong

  /**
   * The total counts for each topic
   */
  @transient private var globalTopicCounter: BDV[Count] = collectGlobalCounter(corpus, numTopics)
  assert(brzSum(globalTopicCounter) == sumTerms)
  @transient private val sc = corpus.vertices.context
  @transient private val seed = new Random().nextInt()
  @transient private var innerIter = 1

  @transient private var cachedEdges: EdgeRDD[ED, VD] = corpus.edges
  @transient private var cachedVertices: VertexRDD[VD] = corpus.vertices

  private def termVertices = corpus.vertices.filter(t => t._1 >= 0)

  private def docVertices = corpus.vertices.filter(t => t._1 < 0)

  private def checkpoint(): Unit = {
    if (innerIter % 10 == 0 && sc.getCheckpointDir.isDefined) {
      val edges = corpus.edges.map(t => t)
      edges.checkpoint()
      val newCorpus: Graph[VD, ED] = Graph.fromEdges(edges, null,
        storageLevel, storageLevel)
      corpus = updateCounter(newCorpus, numTopics).cache()
    }
  }

  private def gibbsSampling(): Unit = {
    val corpusTopicDist = collectTermTopicDist(corpus, globalTopicCounter,
      sumTerms, numTerms, numTopics, alpha, beta)

    val corpusSampleTopics = sampleTopics(corpusTopicDist, globalTopicCounter,
      sumTerms, innerIter + seed, numTerms, numTopics, alpha, beta)
    corpusSampleTopics.edges.setName(s"edges-$innerIter").cache().count()
    Option(cachedEdges).foreach(_.unpersist())
    cachedEdges = corpusSampleTopics.edges

    corpus = updateCounter(corpusSampleTopics, numTopics)
    corpus.vertices.setName(s"vertices-$innerIter").cache()
    globalTopicCounter = collectGlobalCounter(corpus, numTopics)
    assert(brzSum(globalTopicCounter) == sumTerms)
    Option(cachedVertices).foreach(_.unpersist())
    cachedVertices = corpus.vertices

    checkpoint()
    innerIter += 1
  }

  def saveTopicModel(burnInIter: Int): TopicModel = {
    val topicModel = TopicModel(numTopics, numTerms, alpha, beta)
    for (iter <- 1 to burnInIter) {
      logInfo("Save TopicModel (Iteration %d/%d)".format(iter, burnInIter))
      gibbsSampling()
      updateTopicModel(termVertices, topicModel)
    }
    topicModel._topicCounter :/= burnInIter.toDouble
    topicModel._topicTermCounter.foreach(_ :/= burnInIter.toDouble)
    topicModel
  }

  def runGibbsSampling(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      logInfo("Start Gibbs sampling (Iteration %d/%d)".format(iter, iterations))
      gibbsSampling()
    }
  }

  @Experimental
  def mergeDuplicateTopic(threshold: Double = 0.95D): Map[Int, Int] = {
    val rows = termVertices.map(t => t._2.counter).map { bsv =>
      val length = bsv.length
      val used = bsv.used
      val index = bsv.index.slice(0, used)
      val data = bsv.data.slice(0, used).map(_.toDouble)
      new SSV(length, index, data).asInstanceOf[SV]
    }
    val simMatrix = new RowMatrix(rows).columnSimilarities()
    val minMap = simMatrix.entries.filter { case MatrixEntry(row, column, sim) =>
      sim > threshold && row != column
    }.map { case MatrixEntry(row, column, sim) =>
      (column.toInt, row.toInt)
    }.groupByKey().map { case (topic, simTopics) =>
      (topic, simTopics.min)
    }.collect().toMap
    if (minMap.size > 0) {
      corpus = corpus.mapEdges(edges => {
        edges.attr.map { topic =>
          minMap.get(topic).getOrElse(topic)
        }
      })
      corpus = updateCounter(corpus, numTopics)
    }
    minMap
  }

  def perplexity(): Double = {
    val totalTopicCounter = this.globalTopicCounter
    val numTopics = this.numTopics
    val numTerms = this.numTerms
    val alpha = this.alpha
    val beta = this.beta

    val newCounts = corpus.mapReduceTriplets[Int](triplet => {
      val size = triplet.attr.size
      val docId = triplet.dstId
      val wordId = triplet.srcId
      Iterator((docId, size), (wordId, size))
    }, (a, b) => a + b)
    val (termProb, totalNum) = corpus.outerJoinVertices(newCounts) {
      (_, f, n) =>
        (f.counter, n.get)
    }.mapTriplets {
      triplet =>
        val (termCounter, _) = triplet.srcAttr
        val (docTopicCounter, docTopicCount) = triplet.dstAttr
        var probWord = 0D
        val size = triplet.attr.size
        (0 until numTopics).foreach {
          topic =>
            val phi = (termCounter(topic) + beta) / (totalTopicCounter(topic) + numTerms * beta)
            val theta = (docTopicCounter(topic) + alpha) / (docTopicCount + alpha * numTopics)
            probWord += phi * theta
        }
        (Math.log(probWord * size) * size, size)
    }.edges.map(t => t.attr).reduce {
      (lhs, rhs) =>
        (lhs._1 + rhs._1, lhs._2 + rhs._2)
    }
    math.exp(-1 * termProb / totalNum)
  }
}


object TopicModeling {


  type DocId = VertexId
  type WordId = VertexId
  type Count = Int

  case class VD(counter: BSV[Count], dist: BSV[Double], dist1: BSV[Double])

  type ED = Array[Count]

  def train(docs: RDD[(DocId, SSV)],
    numTopics: Int = 2048,
    totalIter: Int = 150,
    burnIn: Int = 5,
    alpha: Double = 0.1,
    beta: Double = 0.01): TopicModel = {
    require(totalIter > burnIn, "totalIter is less than burnIn")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    val topicModeling = new TopicModeling(docs, numTopics, alpha, beta)
    topicModeling.runGibbsSampling(totalIter - burnIn)
    topicModeling.saveTopicModel(burnIn)
  }

  def incrementalTrain(docs: RDD[(DocId, SSV)],
    computedModel: TopicModel,
    totalIter: Int = 150,
    burnIn: Int = 5): TopicModel = {
    require(totalIter > burnIn, "totalIter is less than burnIn")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    val numTopics = computedModel.topicCounter.size
    val alpha = computedModel.alpha
    val beta = computedModel.beta

    val broadcastModel = docs.context.broadcast(computedModel)
    val topicModeling = new TopicModeling(docs, numTopics, alpha, beta,
      computedModel = broadcastModel)
    broadcastModel.unpersist()
    topicModeling.runGibbsSampling(totalIter - burnIn)
    topicModeling.saveTopicModel(burnIn)
  }

  @inline private[mllib] def maxMinIndexSearch[V](v: BSV[V], i: Int,
    lastReturnedPos: Int): Int = {
    val array = v.array
    val index = array.index
    if (array.activeSize == 0) return -1
    if (index(0) > i) return -1
    if (lastReturnedPos >= array.activeSize - 1) return array.activeSize - 1
    var begin = lastReturnedPos
    var end = array.activeSize - 1
    var found = false
    if (end > i) end = i
    if (begin < 0) begin = 0

    var mid = (end + begin) >> 1

    while (!found && begin <= end) {
      if (index(mid) < i) {
        begin = mid + 1
        mid = (end + begin) >> 1
      }
      else if (index(mid) > i) {
        end = mid - 1
        mid = (end + begin) >> 1
      }
      else {
        found = true
      }
    }

    if (found || index(mid) < i || mid == 0) {
      mid
    }
    else {
      mid - 1
    }
  }

  private[mllib] def collectTermTopicDist(graph: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    sumTerms: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double): Graph[VD, ED] = {
    val newVD = graph.vertices.filter(_._1 >= 0).map { v =>
      val vertexId = v._1
      val termTopicCounter = v._2.counter
      termTopicCounter.compact()
      val length = termTopicCounter.length
      val used = termTopicCounter.used
      val index = termTopicCounter.index
      val data = termTopicCounter.data
      val w = new Array[Double](used)
      val w1 = new Array[Double](used)

      var wi = 0D
      var i = 0

      while (i < used) {
        val topic = index(i)
        val count = data(i)
        var adjustment = 0D
        val alphaAS = alpha

        w(i) = count * ((totalTopicCounter(topic) * (alpha * numTopics)) +
          (alpha * numTopics) * (adjustment + alphaAS) +
          adjustment * (sumTerms - 1 + (alphaAS * numTopics))) /
          (totalTopicCounter(topic) + (numTerms * beta)) /
          (sumTerms - 1 + (alphaAS * numTopics))

        adjustment = -1D
        w1(i) = count * ((totalTopicCounter(topic) * (alpha * numTopics)) +
          (alpha * numTopics) * (adjustment + alphaAS) +
          adjustment * (sumTerms - 1 + (alphaAS * numTopics))) /
          (totalTopicCounter(topic) + (numTerms * beta)) /
          (sumTerms - 1 + (alphaAS * numTopics))

        w1(i) = w1(i) - w(i)
        wi = w(i) + wi
        w(i) = wi
        i += 1
      }
      val vd = VD(termTopicCounter, new BSV[Double](index, w, used, length),
        new BSV[Double](index, w1, used, length))
      (vertexId, vd)
    }
    graph.joinVertices(newVD)((_, _, nvd) => nvd)
  }

  @inline private def collectDocTopicDist(
    totalTopicCounter: BDV[Count],
    termTopicCounter: BSV[Count],
    docTopicCounter: BSV[Count],
    d: BDV[Double],
    d1: BDV[Double],
    sumTerms: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double): Unit = {
    val used = docTopicCounter.used
    val index = docTopicCounter.index
    val data = docTopicCounter.data

    var i = 0
    var di = 0D
    while (i < used) {
      val topic = index(i)
      val count = data(i)
      var adjustment = 0D
      val alphaAS = alpha

      d(topic) = count * (sumTerms - 1 + (alphaAS * numTopics)) *
        (termTopicCounter(topic) + (adjustment + beta)) /
        (totalTopicCounter(topic) + adjustment + numTerms * beta) /
        (sumTerms - 1 + (alphaAS * numTopics))

      adjustment = -1D
      d1(topic) = count * (sumTerms - 1 + (alphaAS * numTopics)) *
        (termTopicCounter(topic) + (adjustment + beta)) /
        (totalTopicCounter(topic) + adjustment + numTerms * beta) /
        (sumTerms - 1 + (alphaAS * numTopics))

      d1(topic) = d1(topic) - d(topic)
      di = d(topic) + di
      d(topic) = di

      i += 1
    }
    d(numTopics - 1) = di
  }

  private def collectGlobalTopicDist(totalTopicCounter: BDV[Count],
    sumTerms: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double): (BDV[Double], BDV[Double]) = {
    var topic = 0
    val t = BDV.zeros[Double](numTopics)
    val t1 = BDV.zeros[Double](numTopics)
    var ti = 0D

    while (topic < numTopics) {
      var adjustment = 0D
      val alphaAS = alpha
      t(topic) = (adjustment + beta) * (totalTopicCounter(topic) * (alpha * numTopics) +
        alpha * numTopics * (adjustment + alphaAS) +
        adjustment * (sumTerms - 1 + (alphaAS * numTopics))) /
        (totalTopicCounter(topic) + (adjustment + numTerms * beta)) /
        (sumTerms - 1 + (alphaAS * numTopics))

      adjustment = -1D
      t1(topic) = (adjustment + beta) * (totalTopicCounter(topic) * (alpha * numTopics) +
        alpha * numTopics * (adjustment + alphaAS) +
        adjustment * (sumTerms - 1 + (alphaAS * numTopics))) /
        (totalTopicCounter(topic) + (adjustment + numTerms * beta)) /
        (sumTerms - 1 + (alphaAS * numTopics))

      t1(topic) = t1(topic) - t(topic)
      ti = t(topic) + ti
      t(topic) = ti

      topic += 1
    }
    (t, t1)
  }

  // scalastyle:off
  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   * Asymmetric Dirichlet Priors you can refer to the paper:
   * "Rethinking LDA: Why Priors Matter", available at
   * [[http://people.ee.duke.edu/~lcarin/Eric3.5.2010.pdf]]
   *
   * if you want to know more about the above codes, you can refer to the following formula:
   * First), the original Gibbis sampling formula is :<img src="http://www.forkosh.com/mathtex.cgi? P(z^{(d)}_{n}|W, Z_{\backslash d,n}, \alpha u, \beta u)\propto P(w^{(d)}_{n}|z^{(d)}_{n},W_{\backslash d,n}, Z_{\backslash d,n}, \beta u) P(z^{(d)}_{n}|Z_{\backslash d,n}, \alpha u)"> (1)
   * Second), using the Asymmetric Dirichlet Priors, the second term of formula (1) can be written as following:
   * <img src="http://www.forkosh.com/mathtex.cgi? P(z^{(d)}_{N_{d+1}}=t|Z, \alpha, \alpha^{'}u)=\int dm P(z^{(d)}_{N_{d+1}}=t|Z, \alpha m)P(m|Z, \alpha^{'}u)=\frac{N_{t|d}+\alpha \frac{\widehat{N}_{T}+\frac{\alpha^{'}}{T}}{\Sigma_{t}\widehat{N}_{t}+ \alpha ^{'}}}{N_{d}+\alpha}"> (2)
   * Third), in this code, we set the <img src="http://www.forkosh.com/mathtex.cgi? \alpha=\alpha^{'}">, you can set different value for them. Additionally, in our code the parameter "alpha" is equal to <img src="http://www.forkosh.com/mathtex.cgi?\alpha * T">;
   * "adjustment" denote that if this is the current topic, you need to reduce number one from the corresponding term;
   * <img src="http://www.forkosh.com/mathtex.cgi? ratio=\frac{\widehat{N}_{t}+\frac{\alpha^{'}}{T}}{\Sigma _{t}\widehat{N}_{t}+\alpha^{'}} \qquad asPrior = ratio * (alpha * numTopics)">;
   * Finally), we put them into formula (1) to get the final Asymmetric Dirichlet Priors Gibbs sampling formula.
   *
   */
  // scalastyle:on
  private[mllib] def sampleTopics(
    graph: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    sumTerms: Long,
    innerIter: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double
  ): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    val (t, t1) = collectGlobalTopicDist(totalTopicCounter, sumTerms, numTerms,
      numTopics, alpha, beta)
    val sampleTopics = (gen: Random, d: BDV[Double], d1: BDV[Double],
    triplet: EdgeTriplet[VD, ED]) => {
      assert(triplet.srcId >= 0)
      assert(triplet.dstId < 0)
      val termTopicCounter = triplet.srcAttr.counter
      val docTopicCounter = triplet.dstAttr.counter
      val topics = triplet.attr
      val w = triplet.srcAttr.dist
      val w1 = triplet.srcAttr.dist1
      collectDocTopicDist(totalTopicCounter, termTopicCounter, docTopicCounter,
        d, d1, sumTerms, numTerms, numTopics, alpha, beta)

      var i = 0
      while (i < topics.length) {
        val currentTopic = topics(i)
        val newTopic = multinomialDistSampler(gen, docTopicCounter, d, w, t,
          d1(currentTopic), w1(currentTopic), t1(currentTopic), currentTopic)
        topics(i) = newTopic
        i += 1
      }
      topics
    }

    graph.mapTriplets {
      (pid, iter) =>
        val gen = new Random(parts * innerIter + pid)
        val d = BDV.zeros[Double](numTopics)
        val d1 = BDV.zeros[Double](numTopics)
        iter.map {
          token =>
            sampleTopics(gen, d, d1, token)
        }
    }
  }

  private[mllib] def updateCounter(graph: Graph[VD, ED], numTopics: Int): Graph[VD, ED] = {
    val newCounter = graph.mapReduceTriplets[BSV[Count]](e => {
      val docId = e.dstId
      val wordId = e.srcId
      val newTopics = e.attr
      val vector = BSV.zeros[Count](numTopics)
      var i = 0
      while (i < newTopics.length) {
        val newTopic = newTopics(i)
        vector(newTopic) += 1
        i += 1
      }
      Iterator((docId, vector), (wordId, vector))

    }, _ :+ _)
    graph.joinVertices(newCounter)((_, _, n) => VD(n, null, null))
  }

  private[mllib] def collectGlobalCounter(graph: Graph[VD, ED],
    numTopics: Int): BDV[Count] = {
    graph.vertices.filter(t => t._1 >= 0).map(_._2.counter).
      aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
  }

  private def updateTopicModel(termVertices: VertexRDD[VD], topicModel: TopicModel): Unit = {
    termVertices.map(t => (t._1.toInt, t._2.counter)).collect().foreach { case (term, ttc) =>
      ttc.activeIterator.foreach {
        case (topic, inc) =>
          topicModel.update(term, topic, inc)
      }
    }
  }

  @inline private def maxMinD(i: Int, docTopicCounter: BSV[Count], d: BDV[Double]) = {
    val lastReturnedPos = maxMinIndexSearch(docTopicCounter, i, -1)
    if (lastReturnedPos > -1) {
      d(docTopicCounter.index(lastReturnedPos))
    }
    else {
      0D
    }
  }

  @inline private def maxMinW(i: Int, w: BSV[Double]) = {
    val lastReturnedPos = maxMinIndexSearch(w, i, -1)
    if (lastReturnedPos > -1) {
      w.data(lastReturnedPos)
    }
    else {
      0D
    }
  }

  @inline private def maxMinT(i: Int, t: BDV[Double]) = {
    t(i)
  }

  @inline private def index(i: Int, docTopicCounter: BSV[Count],
    d: BDV[Double], w: BSV[Double], t: BDV[Double],
    d1: Double, w1: Double, t1: Double, currentTopic: Int) = {
    val lastDS = maxMinD(i, docTopicCounter, d)
    val lastWS = maxMinW(i, w)
    val lastTS = maxMinT(i, t)
    if (i >= currentTopic) {
      lastDS + lastWS + lastTS + d1 + w1 + t1
    } else {
      lastDS + lastWS + lastTS
    }
  }

  @inline private def multinomialDistSampler[V](rand: Random, docTopicCounter: BSV[Count],
    d: BDV[Double], w: BSV[Double], t: BDV[Double],
    d1: Double, w1: Double, t1: Double, currentTopic: Int): Int = {
    val numTopics = d.length
    val distSum = rand.nextDouble() * (t(numTopics - 1) + t1 +
      w.data(w.used - 1) + w1 + d(numTopics - 1) + d1)

    var begin = 0
    var end = numTopics - 1
    var found = false
    var mid = (end + begin) >> 1
    var sum = 0D
    var isLeft = false

    while (!found && begin <= end) {
      sum = index(mid, docTopicCounter, d, w, t, d1, w1, t1, currentTopic)
      if (sum < distSum) {
        isLeft = false
        begin = mid + 1
        mid = (end + begin) >> 1
      }
      else if (sum > distSum) {
        isLeft = true
        end = mid - 1
        mid = (end + begin) >> 1
      }
      else {
        found = true
      }
    }
    val topic = if (sum < distSum && mid < numTopics - 1) {
      mid + 1
    }
    else if (isLeft) {
      mid + 1
    } else {
      mid - 1
    }

    assert(index(topic, docTopicCounter, d, w, t, d1, w1, t1, currentTopic) >= distSum)
    if (topic > 0) {
      assert(index(topic - 1, docTopicCounter, d, w, t, d1, w1, t1, currentTopic) <= distSum)
    }

    topic
  }

  /**
   * A uniform distribution sampler, which is only used for initialization.
   */
  private[mllib] def uniformDistSampler(rand: Random, dimension: Int): Int = {
    rand.nextInt(dimension)
  }

  private[mllib] def initializeCorpus(
    docs: RDD[(TopicModeling.DocId, SSV)],
    numTopics: Int,
    storageLevel: StorageLevel,
    computedModel: Broadcast[TopicModel] = null): Graph[VD, ED] = {
    val edges = docs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      var model: TopicModel = null
      if (computedModel != null) model = computedModel.value
      iter.flatMap {
        case (docId, doc) =>
          initializeEdges(gen, doc, docId, numTopics, model)
      }
    })
    var corpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    corpus.partitionBy(PartitionStrategy.EdgePartition1D)
    corpus = updateCounter(corpus, numTopics).cache()
    corpus.vertices.count()
    corpus
  }

  private def initializeEdges(gen: Random, doc: SSV, docId: DocId, numTopics: Int,
    computedModel: TopicModel = null): Array[Edge[ED]] = {
    assert(docId >= 0)
    val newDocId: DocId = -(docId + 1L)
    val indices = doc.indices
    val values = doc.values
    if (computedModel == null) {
      indices.zip(values).map {
        case (term, counter) =>
          val topic = new Array[Int](counter.toInt)
          for (i <- 0 until counter.toInt) {
            topic(i) = uniformDistSampler(gen, numTopics)
          }
          Edge(term, newDocId, topic)
      }
    }
    else {
      val topics = values.map(i => new Array[Int](i.toInt))
      val docTopicCounter = computedModel.uniformDistSamplerForDocument(indices,
        topics, numTopics, gen)
      (0 to 2).foreach(t => computedModel.generateTopicDistForDocument(
        docTopicCounter, indices, topics, realTime = false, gen))
      indices.zip(topics).map {
        case (term, topic) =>
          Edge(term, newDocId, topic)
      }
    }
  }
}

class TopicModelingKryoRegistrator extends KryoRegistrator {
  def registerClasses(kryo: com.esotericsoftware.kryo.Kryo) {
    val gkr = new GraphKryoRegistrator
    gkr.registerClasses(kryo)

    kryo.register(classOf[BSV[TopicModeling.Count]])
    kryo.register(classOf[BSV[TopicModel.Count]])

    kryo.register(classOf[BDV[TopicModeling.Count]])
    kryo.register(classOf[BDV[TopicModel.Count]])

    kryo.register(classOf[SV])
    kryo.register(classOf[SSV])
    kryo.register(classOf[SDV])

    kryo.register(classOf[TopicModeling.ED])
    kryo.register(classOf[TopicModeling.VD])

    kryo.register(classOf[Random])
    kryo.register(classOf[TopicModeling])
    kryo.register(classOf[TopicModel])
  }
}
