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


  @transient private var globalParameter: GlobalParameter = collectGlobalParameter(corpus)

  @transient private val sc = corpus.vertices.context
  @transient private val seed = new Random().nextInt()
  @transient private var innerIter = 1
  @transient private var cachedEdges: EdgeRDD[ED, _] = corpus.edges
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

  private def collectGlobalParameter(graph: Graph[VD, ED]): GlobalParameter = {
    val globalTopicCounter = collectGlobalCounter(graph, numTopics)
    assert(brzSum(globalTopicCounter) == sumTerms)
    val (denominator, denominator1) = denominatorBDV(globalTopicCounter,
      sumTerms, numTerms, numTopics, alpha, beta)
    val (t, t1) = collectGlobalTopicDist(globalTopicCounter, denominator, denominator1,
      sumTerms, numTopics, alpha, beta)
    GlobalParameter(globalTopicCounter, t, t1, denominator, denominator1)
  }

  private def gibbsSampling(): Unit = {
    val broadcast = sc.broadcast(globalParameter)
    val corpusTopicDist = collectTermTopicDist(corpus, broadcast,
      sumTerms, numTopics, alpha, beta)

    val corpusSampleTopics = sampleTopics(corpusTopicDist, broadcast,
      sumTerms, innerIter + seed, numTopics, alpha, beta)
    corpusSampleTopics.edges.setName(s"edges-$innerIter").cache().count()
    Option(cachedEdges).foreach(_.unpersist())
    cachedEdges = corpusSampleTopics.edges

    corpus = updateCounter(corpusSampleTopics, numTopics)
    corpus.vertices.setName(s"vertices-$innerIter").cache()
    globalParameter = collectGlobalParameter(corpus)
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
    topicModel.gtc :/= burnInIter.toDouble
    topicModel.ttc.foreach(_ :/= burnInIter.toDouble)
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
    val rows = termVertices.map(t => t._2).map { bsv =>
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
    val totalTopicCounter = this.globalParameter.totalTopicCounter
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
      (_, counter, n) =>
        (counter, n.get)
    }.mapTriplets {
      triplet =>
        val (termCounter, _) = triplet.srcAttr
        val (docTopicCounter, docTopicCount) = triplet.dstAttr
        var probWord = 0D
        val size = triplet.attr.size
        (0 until numTopics).foreach {
          topic =>
            val phi = (termCounter(topic) + beta) / (totalTopicCounter(topic) + numTerms * beta)
            val theta = (docTopicCounter(topic) + alpha) / (docTopicCount + numTopics * alpha)
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

  private[mllib] type DocId = VertexId
  private[mllib] type WordId = VertexId
  private[mllib] type Count = Int
  private[mllib] type ED = Array[Count]
  private[mllib] type VD = BSV[Count]

  private[mllib] case class GlobalParameter(totalTopicCounter: BDV[Count],
    t: BDV[Double], t1: BDV[Double], denominator: BDV[Double], denominator1: BDV[Double])

  private[mllib] case class Parameter(counter: BSV[Count], dist: BSV[Double], dist1: BSV[Double])

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
    val numTopics = computedModel.ttc.size
    val alpha = computedModel.alpha
    val beta = computedModel.beta

    val broadcastModel = docs.context.broadcast(computedModel)
    val topicModeling = new TopicModeling(docs, numTopics, alpha, beta,
      computedModel = broadcastModel)
    broadcastModel.unpersist()
    topicModeling.runGibbsSampling(totalIter - burnIn)
    topicModeling.saveTopicModel(burnIn)
  }

  @inline private def collectDocTopicDist(
    denominator: BDV[Double],
    denominator1: BDV[Double],
    termTopicCounter: BSV[Count],
    docTopicCounter: BSV[Count],
    d: BDV[Double],
    d1: BDV[Double],
    sumTerms: Long,
    numTopics: Int,
    alpha: Double,
    beta: Double): Unit = {
    val used = docTopicCounter.used
    val index = docTopicCounter.index
    val data = docTopicCounter.data

    val alphaAS = alpha
    val termSum = sumTerms - 1D + alphaAS * numTopics

    var i = 0
    var di = 0D

    while (i < used) {
      val topic = index(i)
      val count = data(i)
      d(topic) = count * termSum * (termTopicCounter(topic) + beta) /
        denominator(topic)

      d1(topic) = count * termSum * (termTopicCounter(topic) - 1D + beta) /
        denominator1(topic)

      d1(topic) = d1(topic) - d(topic)
      di = d(topic) + di
      d(topic) = di

      i += 1
    }
    d(numTopics - 1) = di
  }

  private[mllib] def collectTermTopicDist(graph: Graph[VD, ED],
    broadcast: Broadcast[GlobalParameter],
    sumTerms: Long,
    numTopics: Int,
    alpha: Double,
    beta: Double): Graph[Parameter, ED] = {
    graph.mapVertices { (vertexId, counter) =>
      val GlobalParameter(totalTopicCounter, _, _, denominator, denominator1) = broadcast.value
      val alphaAS = alpha
      val alphaSum = alpha * numTopics
      val termSum = sumTerms - 1D + alphaAS * numTopics
      if (vertexId >= 0) {
        val termTopicCounter = counter
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
          w(i) = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
            denominator(topic)

          w1(i) = count * (alphaSum * (totalTopicCounter(topic) - 1D + alphaAS) - termSum) /
            denominator1(topic)

          w1(i) = w1(i) - w(i)
          wi = w(i) + wi
          w(i) = wi
          i += 1
        }
        Parameter(termTopicCounter, new BSV[Double](index, w, used, length),
          new BSV[Double](index, w1, used, length))
      }
      else {
        val docTopicCounter = counter
        docTopicCounter.compact()
        Parameter(docTopicCounter, null, null)
      }
    }
  }

  private def collectGlobalTopicDist(
    totalTopicCounter: BDV[Count],
    denominator: BDV[Double],
    denominator1: BDV[Double],
    sumTerms: Long,
    numTopics: Int,
    alpha: Double,
    beta: Double): (BDV[Double], BDV[Double]) = {
    val t = BDV.zeros[Double](numTopics)
    val t1 = BDV.zeros[Double](numTopics)

    val alphaAS = alpha
    val alphaSum = alpha * numTopics
    val termSum = sumTerms - 1D + alphaAS * numTopics

    var ti = 0D
    for (topic <- 0 until numTopics) {
      t(topic) = beta * (totalTopicCounter(topic) * alphaSum +
        alphaSum * alphaAS) / denominator(topic)

      t1(topic) = (-1D + beta) * (totalTopicCounter(topic) * alphaSum +
        alphaSum * (-1D + alphaAS) - termSum) / denominator1(topic)

      t1(topic) = t1(topic) - t(topic)
      ti = t(topic) + ti
      t(topic) = ti
    }
    (t, t1)
  }

  private def denominatorBDV(
    totalTopicCounter: BDV[Count],
    sumTerms: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double): (BDV[Double], BDV[Double]) = {
    val alphaAS = alpha
    val termSum = sumTerms - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    val denominator = BDV.zeros[Double](numTopics)
    val denominator1 = BDV.zeros[Double](numTopics)
    for (topic <- 0 until numTopics) {
      denominator(topic) = totalTopicCounter(topic) + betaSum
      denominator(topic) = denominator(topic) * termSum

      denominator1(topic) = totalTopicCounter(topic) - 1D + betaSum
      denominator1(topic) = denominator1(topic) * termSum
    }
    (denominator, denominator1)
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
    graph: Graph[Parameter, ED],
    broadcast: Broadcast[GlobalParameter],
    sumTerms: Long,
    innerIter: Long,
    numTopics: Int,
    alpha: Double,
    beta: Double): Graph[Parameter, ED] = {
    val parts = graph.edges.partitions.size

    val sampleTopics = (gen: Random,
    totalTopicCounter: BDV[Count],
    t: BDV[Double],
    t1: BDV[Double],
    denominator: BDV[Double],
    denominator1: BDV[Double],
    d: BDV[Double],
    d1: BDV[Double],
    triplet: EdgeTriplet[Parameter, ED]) => {
      assert(triplet.srcId >= 0)
      assert(triplet.dstId < 0)
      val termTopicCounter = triplet.srcAttr.counter
      val docTopicCounter = triplet.dstAttr.counter
      val topics = triplet.attr
      val w = triplet.srcAttr.dist
      val w1 = triplet.srcAttr.dist1
      collectDocTopicDist(denominator, denominator1, termTopicCounter, docTopicCounter,
        d, d1, sumTerms, numTopics, alpha, beta)

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
        val GlobalParameter(totalTopicCounter, t, t1, denominator, denominator1) = broadcast.value

        iter.map {
          token =>
            sampleTopics(gen, totalTopicCounter, t, t1, denominator, denominator1, d, d1, token)
        }
    }
  }

  private def updateCounter[D](graph: Graph[D, ED], numTopics: Int): Graph[VD, ED] = {
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
    graph.outerJoinVertices(newCounter)((_, _, n) => n.get)
  }

  private def collectGlobalCounter(graph: Graph[VD, ED], numTopics: Int): BDV[Count] = {
    graph.vertices.filter(t => t._1 >= 0).map(_._2).
      aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
  }

  private def updateTopicModel(termVertices: VertexRDD[VD], topicModel: TopicModel): Unit = {
    termVertices.map(vertex => {
      val termTopicCounter = vertex._2
      val index = termTopicCounter.index.slice(0, termTopicCounter.used)
      val data = termTopicCounter.data.slice(0, termTopicCounter.used).map(_.toDouble)
      val used = termTopicCounter.used
      val length = termTopicCounter.length
      (vertex._1.toInt, new BSV[Double](index, data, used, length))
    }).collect().foreach { case (term, counter) =>
      topicModel.merge(term, counter)
    }
  }

  private def initializeCorpus(
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
    //corpus.partitionBy(PartitionStrategy.EdgePartition1D)
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
        docTopicCounter, indices, topics, gen))
      indices.zip(topics).map {
        case (term, topic) =>
          Edge(term, newDocId, topic)
      }
    }
  }

  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   */
  @inline private def multinomialDistSampler[V](rand: Random, docTopicCounter: BSV[Count],
    d: BDV[Double], w: BSV[Double], t: BDV[Double],
    d1: Double, w1: Double, t1: Double, currentTopic: Int): Int = {
    val numTopics = d.length
    val distSum = rand.nextDouble() * (t(numTopics - 1) + t1 +
      w.data(w.used - 1) + w1 + d(numTopics - 1) + d1)
    val fun = index(docTopicCounter, d, w, t, d1, w1, t1, currentTopic) _
    minMaxValueSearch(fun, distSum, numTopics)
  }

  /**
   * A uniform distribution sampler, which is only used for initialization.
   */
  @inline private[mllib] def uniformDistSampler(rand: Random, dimension: Int): Int = {
    rand.nextInt(dimension)
  }

  @inline private[mllib] def minMaxIndexSearch[V](v: BSV[V], i: Int,
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

    val minMax = if (found || index(mid) < i || mid == 0) {
      mid
    }
    else {
      mid - 1
    }
    // assert(index(minMax) <= i)
    // if (minMax < array.activeSize - 1) assert(index(minMax + 1) > i)
    minMax
  }

  @inline private[mllib] def minMaxValueSearch(index: (Int) => Double, distSum: Double,
    numTopics: Int): Int = {
    var begin = 0
    var end = numTopics - 1
    var found = false
    var mid = (end + begin) >> 1
    var sum = 0D
    var isLeft = false
    while (!found && begin <= end) {
      sum = index(mid)
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
    val topic = if (sum < distSum) {
      mid + 1
    }
    else if (isLeft) {
      mid + 1
    } else {
      mid - 1
    }
    // assert(index(topic) >= distSum)
    // if (topic > 0) assert(index(topic - 1) <= distSum)
    topic
  }

  @inline private def maxMinD(i: Int, docTopicCounter: BSV[Count], d: BDV[Double]) = {
    val lastReturnedPos = minMaxIndexSearch(docTopicCounter, i, -1)
    if (lastReturnedPos > -1) {
      d(docTopicCounter.index(lastReturnedPos))
    }
    else {
      0D
    }
  }

  @inline private def maxMinW(i: Int, w: BSV[Double]) = {
    val lastReturnedPos = minMaxIndexSearch(w, i, -1)
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

  @inline private def index(docTopicCounter: BSV[Count],
    d: BDV[Double], w: BSV[Double], t: BDV[Double],
    d1: Double, w1: Double, t1: Double, currentTopic: Int)(i: Int) = {
    val lastDS = maxMinD(i, docTopicCounter, d)
    val lastWS = maxMinW(i, w)
    val lastTS = maxMinT(i, t)
    if (i >= currentTopic) {
      lastDS + lastWS + lastTS + d1 + w1 + t1
    } else {
      lastDS + lastWS + lastTS
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
    kryo.register(classOf[TopicModeling.GlobalParameter])

    kryo.register(classOf[Random])
    kryo.register(classOf[TopicModeling])
    kryo.register(classOf[TopicModel])
  }
}
