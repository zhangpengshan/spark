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

package org.apache.spark.mllib.clustering

import java.util.Random

import scala.collection.mutable
import scala.reflect.ClassTag

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx._
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext._
import org.apache.spark.util.random.XORShiftRandom

import LDA._

class LDA private[mllib](
  @transient var corpus: Graph[VD, ED],
  val numTopics: Int,
  val numTerms: Int,
  val alpha: Double,
  val beta: Double,
  val alphaAS: Double,
  @transient val storageLevel: StorageLevel)
  extends Serializable with Logging {

  def this(docs: RDD[(DocId, SSV)],
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    computedModel: Broadcast[LDAModel] = null) {
    this(initializeCorpus(docs, numTopics, storageLevel, computedModel),
      numTopics, docs.first()._2.size, alpha, beta, alphaAS, storageLevel)
  }

  /**
   * The number of documents in the corpus
   */
  val numDocs = docVertices.count()

  /**
   * The number of token in the corpus
   */
  private val numTokens = corpus.edges.map(e => e.attr.size.toDouble).sum().toLong

  @transient private val sc = corpus.vertices.context
  @transient private val seed = new Random().nextInt()
  @transient private var innerIter = 1
  @transient private var totalTopicCounter: BDV[Count] = collectTotalTopicCounter(corpus)

  private def termVertices = corpus.vertices.filter(t => t._1 >= 0)

  private def docVertices = corpus.vertices.filter(t => t._1 < 0)

  private def checkpoint(): Unit = {
    if (innerIter % 10 == 0 && sc.getCheckpointDir.isDefined) {
      val edges = corpus.edges.map(t => t)
      edges.checkpoint()
      val newCorpus: Graph[VD, ED] = Graph.fromEdges(edges, null,
        storageLevel, storageLevel)
      corpus = updateCounter(newCorpus, numTopics).persist(storageLevel)
    }
  }

  private def collectTotalTopicCounter(graph: Graph[VD, ED]): BDV[Count] = {
    val globalTopicCounter = collectGlobalCounter(graph, numTopics)
    assert(brzSum(globalTopicCounter) == numTokens)
    globalTopicCounter
  }

  private def gibbsSampling(): Unit = {
    val sampleCorpus = sampleTokens(corpus, totalTopicCounter,
      innerIter + seed, numTokens, numTerms, numTopics, alpha, beta, alphaAS)
    sampleCorpus.persist(storageLevel)

    val counterCorpus = updateCounter(sampleCorpus, numTopics)
    counterCorpus.persist(storageLevel)
    // counterCorpus.vertices.count()
    counterCorpus.edges.count()
    totalTopicCounter = collectTotalTopicCounter(counterCorpus)

    corpus.edges.unpersist(false)
    corpus.vertices.unpersist(false)
    sampleCorpus.edges.unpersist(false)
    sampleCorpus.vertices.unpersist(false)
    corpus = counterCorpus

    checkpoint()
    innerIter += 1
  }

  def saveModel(burnInIter: Int): LDAModel = {
    var termTopicCounter: RDD[(Int, BSV[Double])] = null
    for (iter <- 1 to burnInIter) {
      logInfo(s"Save TopicModel (Iteration $iter/$burnInIter)")
      var previousTermTopicCounter = termTopicCounter
      gibbsSampling()
      val newTermTopicCounter = updateModel(termVertices)
      termTopicCounter = Option(termTopicCounter).map(_.join(newTermTopicCounter).map {
        case (term, (a, b)) =>
          val c = a + b
          c.compact()
          (term, c)
      }).getOrElse(newTermTopicCounter)

      termTopicCounter.cache().count()
      Option(previousTermTopicCounter).foreach(_.unpersist())
      previousTermTopicCounter = termTopicCounter
    }
    val model = LDAModel(numTopics, numTerms, alpha, beta)
    termTopicCounter.collect().foreach { case (term, counter) =>
      model.merge(term, counter)
    }
    model.gtc :/= burnInIter.toDouble
    model.ttc.foreach { ttc =>
      ttc :/= burnInIter.toDouble
      ttc.compact()
    }
    model
  }

  def runGibbsSampling(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      println(s"perplexity $iter: ${perplexity}")
      logInfo(s"Start Gibbs sampling (Iteration $iter/$iterations)")
      gibbsSampling()
    }
  }

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
    val totalTopicCounter = this.totalTopicCounter
    val numTopics = this.numTopics
    val numTerms = this.numTerms
    val alpha = this.alpha
    val beta = this.beta
    val totalSize = brzSum(totalTopicCounter)
    var totalProb = 0D

    totalTopicCounter.activeIterator.foreach { case (topic, cn) =>
      totalProb += alpha * beta / (cn + numTerms * beta)
    }

    val termProb = corpus.mapVertices { (vid, counter) =>
      val probDist = BSV.zeros[Double](numTopics)
      if (vid >= 0) {
        val termTopicCounter = counter
        termTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * alpha /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      } else {
        val docTopicCounter = counter
        docTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * beta /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      }
      probDist.compact()
      (counter, probDist)
    }.mapTriplets { triplet =>
      val (termTopicCounter, termProb) = triplet.srcAttr
      val (docTopicCounter, docProb) = triplet.dstAttr
      val docSize = brzSum(docTopicCounter)
      val docTermSize = triplet.attr.length

      var prob = 0D

      docTopicCounter.activeIterator.foreach { case (topic, cn) =>
        prob += cn * termTopicCounter(topic) /
          (totalTopicCounter(topic) + numTerms * beta)
      }
      prob += brzSum(docProb) + brzSum(termProb) + totalProb
      prob += prob / (docSize + numTopics * alpha)

      docTermSize * Math.log(prob)
    }.edges.map(t => t.attr).sum()

    math.exp(-1 * termProb / totalSize)
  }
}

object LDA {

  import LDAUtils._

  private[mllib] type DocId = VertexId
  private[mllib] type WordId = VertexId
  private[mllib] type Count = Int
  private[mllib] type ED = Array[Count]
  private[mllib] type VD = BSV[Count]

  def train(docs: RDD[(DocId, SSV)],
    numTopics: Int = 2048,
    totalIter: Int = 150,
    burnIn: Int = 5,
    alpha: Double = 0.1,
    beta: Double = 0.01,
    alphaAS: Double = 0.1): LDAModel = {
    require(totalIter > burnIn, "totalIter is less than burnIn")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    val topicModeling = new LDA(docs, numTopics, alpha, beta, alphaAS)
    topicModeling.runGibbsSampling(totalIter - burnIn)
    topicModeling.saveModel(burnIn)
  }

  def incrementalTrain(docs: RDD[(DocId, SSV)],
    computedModel: LDAModel,
    alphaAS: Double = 1,
    totalIter: Int = 150,
    burnIn: Int = 5): LDAModel = {
    require(totalIter > burnIn, "totalIter is less than burnIn")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    val numTopics = computedModel.ttc.size
    val alpha = computedModel.alpha
    val beta = computedModel.beta

    val broadcastModel = docs.context.broadcast(computedModel)
    val topicModeling = new LDA(docs, numTopics, alpha, beta, alphaAS,
      computedModel = broadcastModel)
    broadcastModel.unpersist()
    topicModeling.runGibbsSampling(totalIter - burnIn)
    topicModeling.saveModel(burnIn)
  }

  // scalastyle:off
  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   * Asymmetric Dirichlet Priors you can refer to the paper:
   * "Rethinking LDA: Why Priors Matter", available at
   * [[http://people.ee.duke.edu/~lcarin/Eric3.5.2010.pdf]]
   * the original Gibbis sampling formula is:
   * \frac{{n}_{kw}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta}} \frac{{n}_{kd} ^{-di}+ \bar{\alpha} \frac{{n}_{k}^{-di} + \acute{\alpha}}{\sum{n}_{k} +\bar{\acute{\alpha}}}}{\sum{n}_{kd}^{-di} +\bar{\alpha}}
   */
  // scalastyle:on
  private[mllib] def sampleTokens(
    graph: Graph[VD, ED],
    totalTopicCounter: BDV[Count],
    innerIter: Long,
    numTokens: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    val broadcastW = graph.edges.sparkContext.broadcast(
      mutable.Map[VertexId, (BSV[Double], BSV[Double])]())
    graph.mapTriplets(
      (pid, iter) => {
        val gen = new XORShiftRandom(parts * innerIter + pid)
        val d = BDV.zeros[Double](numTopics)
        val d1 = BDV.zeros[Double](numTopics)
        val wCache = broadcastW.value
        var tT: (BDV[Double], BDV[Double]) = null
        iter.map {
          triplet =>
            val term = triplet.srcId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr
            docTopicCounter.synchronized {
              this.d(totalTopicCounter, termTopicCounter, docTopicCounter,
                d, d1, numTokens, numTerms, numTopics, beta, alphaAS)
            }
            val (w, w1) = termTopicCounter.synchronized {
              wCache.synchronized {
                wCache.getOrElseUpdate(term, this.w(totalTopicCounter, termTopicCounter,
                  numTokens, numTerms, numTopics, alpha, beta, alphaAS))
              }
            }

            if (tT == null) tT = LDA.t(totalTopicCounter, numTokens, numTerms,
              numTopics, alpha, beta, alphaAS)
            val (t, t1) = tT
            var i = 0
            while (i < topics.length) {
              val currentTopic = topics(i)
              val adjustment = d1(currentTopic) + w1(currentTopic) + t1(currentTopic)
              val newTopic = docTopicCounter.synchronized {
                termTopicCounter.synchronized {
                  multinomialDistSampler(gen, docTopicCounter, d, w, t,
                    adjustment, currentTopic)
                }
              }
              assert(newTopic < numTopics)
              if (currentTopic != newTopic) {
                docTopicCounter.synchronized {
                  docTopicCounter(currentTopic) -= 1
                  docTopicCounter(newTopic) += 1
                  if (docTopicCounter(currentTopic) == 0) {
                    docTopicCounter.compact()
                  }
                }
                termTopicCounter.synchronized {
                  termTopicCounter(currentTopic) -= 1
                  termTopicCounter(newTopic) += 1
                  if (termTopicCounter(currentTopic) == 0) {
                    termTopicCounter.compact()
                  }
                }

                totalTopicCounter(currentTopic) -= 1
                totalTopicCounter(newTopic) += 1

                if (gen.nextDouble() < 0.001) {
                  wCache -= term
                  tT = null
                }
              }
              topics(i) = newTopic
              i += 1
            }
            topics
        }
      }, TripletFields.All)
  }

  private def updateCounter(graph: Graph[VD, ED], numTopics: Int): Graph[VD, ED] = {
    val newCounter = graph.aggregateMessages[BSV[Count]](ctx => {
      val topics = ctx.attr
      val vector = BSV.zeros[Count](numTopics)
      for (topic <- topics) {
        vector(topic) += 1
      }
      ctx.sendToDst(vector)
      ctx.sendToSrc(vector)
    }, (a, b) => {
      val c = a + b
      c.compact()
      c
    }, TripletFields.EdgeOnly)
    graph.outerJoinVertices(newCounter)((_, _, n) => n.get)
  }

  private def collectGlobalCounter(graph: Graph[VD, ED], numTopics: Int): BDV[Count] = {
    graph.vertices.filter(t => t._1 >= 0).map(_._2).
      aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
  }

  private def updateModel(termVertices: VertexRDD[VD]): RDD[(Int, BSV[Double])] = {
    termVertices.map(vertex => {
      val termTopicCounter = vertex._2
      val index = termTopicCounter.index.slice(0, termTopicCounter.used)
      val data = termTopicCounter.data.slice(0, termTopicCounter.used).map(_.toDouble)
      val used = termTopicCounter.used
      val length = termTopicCounter.length
      (vertex._1.toInt, new BSV[Double](index, data, used, length))
    })
  }

  private def initializeCorpus(
    docs: RDD[(LDA.DocId, SSV)],
    numTopics: Int,
    storageLevel: StorageLevel,
    computedModel: Broadcast[LDAModel] = null): Graph[VD, ED] = {
    val edges = docs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      var model: LDAModel = null
      if (computedModel != null) model = computedModel.value
      iter.flatMap {
        case (docId, doc) =>
          initializeEdges(gen, new BSV[Int](doc.indices, doc.values.map(_.toInt), doc.size),
            docId, numTopics, model)
      }
    })
    var corpus: Graph[VD, ED] = Graph.fromEdges(edges, null, storageLevel, storageLevel)
    // corpus.partitionBy(PartitionStrategy.EdgePartition1D)
    corpus = updateCounter(corpus, numTopics).cache()
    corpus.vertices.count()
    corpus
  }

  private def initializeEdges(
    gen: Random,
    doc: BSV[Int],
    docId: DocId,
    numTopics: Int,
    computedModel: LDAModel = null): Array[Edge[ED]] = {
    assert(docId >= 0)
    val newDocId: DocId = -(docId + 1L)
    if (computedModel == null) {
      doc.activeIterator.map { case (term, counter) =>
        val ev = (0 until counter).map { i =>
          uniformDistSampler(gen, numTopics)
        }.toArray
        Edge(term, newDocId, ev)
      }.toArray
    }
    else {
      val tokens = computedModel.vec2Array(doc)
      val topics = new Array[Int](tokens.length)
      var docTopicCounter = computedModel.uniformDistSampler(tokens, topics, gen)
      for (t <- 0 until 15) {
        docTopicCounter = computedModel.sampleTokens(docTopicCounter,
          tokens, topics, gen)
      }
      doc.activeIterator.map { case (term, counter) =>
        val ev = topics.zipWithIndex.filter { case (topic, offset) =>
          term == tokens(offset)
        }.map(_._1)
        Edge(term, newDocId, ev)
      }.toArray
    }
  }

  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   */
  @inline private def multinomialDistSampler[V](
    rand: Random,
    docTopicCounter: BSV[Count],
    d: BDV[Double],
    w: BSV[Double],
    t: BDV[Double],
    adjustment: Double,
    currentTopic: Int): Int = {
    val numTopics = d.length
    val lastSum = t(numTopics - 1) + w.data(w.used - 1) + d(numTopics - 1) + adjustment
    val distSum = rand.nextDouble() * lastSum
    val fun = index(docTopicCounter, d, w, t, adjustment, currentTopic) _
    val topic = binarySearchInterval(fun, distSum, 0, numTopics, true)
    math.min(topic, numTopics - 1)
  }

  @inline private def index(
    docTopicCounter: BSV[Count],
    d: BDV[Double],
    w: BSV[Double],
    t: BDV[Double],
    adjustment: Double,
    currentTopic: Int)(i: Int) = {
    val lastDS = maxMinD(i, docTopicCounter, d)
    val lastWS = binarySearchSparseVector(i, w)
    val lastTS = t(i)
    if (i >= currentTopic) {
      lastDS + lastWS + lastTS + adjustment
    } else {
      lastDS + lastWS + lastTS
    }
  }

  @inline private def d(
    totalTopicCounter: BDV[Count],
    termTopicCounter: BSV[Count],
    docTopicCounter: BSV[Count],
    d: BDV[Double],
    d1: BDV[Double],
    numTokens: Long,
    numTerms: Int,
    numTopics: Int,
    beta: Double,
    alphaAS: Double): Unit = {
    val used = docTopicCounter.used
    val index = docTopicCounter.index
    val data = docTopicCounter.data

    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var i = 0
    var lastSum = 0D

    while (i < used) {
      val topic = index(i)
      val count = data(i)
      val lastD = count * termSum * (termTopicCounter(topic) + beta) /
        ((totalTopicCounter(topic) + betaSum) * termSum)

      val lastD1 = count * termSum * (termTopicCounter(topic) - 1D + beta) /
        ((totalTopicCounter(topic) - 1D + betaSum) * termSum)

      lastSum += lastD
      d(topic) = lastSum
      d1(topic) = lastD1 - lastD

      i += 1
    }
    d(numTopics - 1) = lastSum
  }

  @inline private def w(
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTokens: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double): (BSV[Double], BSV[Double]) = {
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta

    val length = termTopicCounter.length
    val used = termTopicCounter.used
    val index = termTopicCounter.index.slice(0, used)
    val data = termTopicCounter.data
    val w = new Array[Double](used)
    val w1 = new Array[Double](used)

    var lastsum = 0D
    var i = 0

    while (i < used) {
      val topic = index(i)
      val count = data(i)
      val lastW = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      val lastW1 = count * (alphaSum * (totalTopicCounter(topic) - 1D + alphaAS) - termSum) /
        ((totalTopicCounter(topic) - 1D + betaSum) * termSum)
      lastsum += lastW
      w(i) = lastsum
      w1(i) = lastW1 - lastW
      i += 1
    }
    (new BSV[Double](index, w, used, length),
      new BSV[Double](index, w1, used, length))
  }

  private def t(
    totalTopicCounter: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    numTopics: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double): (BDV[Double], BDV[Double]) = {
    val t = BDV.zeros[Double](numTopics)
    val t1 = BDV.zeros[Double](numTopics)

    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var lastSum = 0D
    for (topic <- 0 until numTopics) {
      val lastT = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)

      val lastT1 = (-1D + beta) * (alphaSum * (totalTopicCounter(topic) +
        (-1D + alphaAS)) - termSum) / ((totalTopicCounter(topic) - 1D + betaSum) * termSum)

      lastSum += lastT
      t(topic) = lastSum
      t1(topic) = lastT1 - lastT
    }
    (t, t1)
  }
}

object LDAUtils {

  /**
   * A uniform distribution sampler, which is only used for initialization.
   */
  @inline private[mllib] def uniformDistSampler(rand: Random, dimension: Int): Int = {
    rand.nextInt(dimension)
  }


  def binarySearchArray[K](
    index: Array[K],
    key: K,
    begin: Int,
    end: Int,
    greater: Boolean)(implicit ord: Ordering[K], ctag: ClassTag[K]): Int = {
    binarySearchInterval(i => index(i), key, begin, end, greater)
  }

  def binarySearchInterval[K](
    index: Int => K,
    key: K,
    begin: Int,
    end: Int,
    greater: Boolean)(implicit ord: Ordering[K], ctag: ClassTag[K]): Int = {
    if (begin == end) {
      return if (greater) end else begin - 1
    }
    var b = begin
    var e = end - 1

    var mid: Int = (e + b) >> 1
    while (b <= e) {
      mid = (e + b) >> 1
      if (ord.lt(index(mid), key)) {
        b = mid + 1
      }
      else if (ord.gt(index(mid), key)) {
        e = mid - 1
      }
      else {
        return mid
      }
    }
    mid = if ((greater && ord.gteq(index(mid), key)) || (!greater && ord.lteq(index(mid), key))) {
      mid
    }
    else if (greater) {
      mid + 1
    }
    else {
      mid - 1
    }
    if (greater) {
      if (mid < end) assert(ord.gteq(index(mid), key))
      if (mid > 0) assert(ord.lteq(index(mid - 1), key))
    } else {
      if (mid > 0) assert(ord.lteq(index(mid), key))
      if (mid < end - 1) assert(ord.gteq(index(mid + 1), key))
    }
    mid
  }

  @inline private[mllib] def binarySearchSparseVector(index: Int, w: BSV[Double]) = {
    val pos = binarySearchArray(w.index, index, 0, w.used, false)
    if (pos > -1) {
      w.data(pos)
    }
    else {
      0D
    }
  }

  @inline private[mllib] def maxMinD[V](
    index: Int,
    docTopicCounter: BSV[V],
    d: BDV[Double]) = {
    val pos = binarySearchArray(docTopicCounter.index, index, 0, docTopicCounter.used, false)
    if (pos > -1) {
      d(docTopicCounter.index(pos))
    }
    else {
      0D
    }
  }


  @inline private[mllib] def maxMinT(i: Int, t: BDV[Double]) = {
    t(i)
  }
}

class LDAKryoRegistrator extends KryoRegistrator {
  def registerClasses(kryo: com.esotericsoftware.kryo.Kryo) {
    val gkr = new GraphKryoRegistrator
    gkr.registerClasses(kryo)

    kryo.register(classOf[BSV[LDA.Count]])
    kryo.register(classOf[BSV[Double]])

    kryo.register(classOf[BDV[LDA.Count]])
    kryo.register(classOf[BDV[Double]])

    kryo.register(classOf[SV])
    kryo.register(classOf[SSV])
    kryo.register(classOf[SDV])

    kryo.register(classOf[LDA.ED])
    kryo.register(classOf[LDA.VD])

    kryo.register(classOf[Random])
    kryo.register(classOf[LDA])
    kryo.register(classOf[LDAModel])
  }
}
