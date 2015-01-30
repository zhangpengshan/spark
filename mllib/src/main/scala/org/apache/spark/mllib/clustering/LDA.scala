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
   * 语料库文档数
   */
  val numDocs = docVertices.count()

  /**
   * 语料库总的词数(包含重复)
   */
  private val sumToken = corpus.edges.map(e => e.attr.size.toDouble).sum().toLong

  @transient private val sc = corpus.vertices.context
  @transient private val seed = new Random().nextInt()
  @transient private var innerIter = 1
  @transient private var globalParameter: GlobalParameter = collectGlobalParameter(corpus)

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

  private def collectGlobalParameter(graph: Graph[VD, ED]): GlobalParameter = {
    val globalTopicCounter = collectGlobalCounter(graph, numTopics)
    assert(brzSum(globalTopicCounter) == sumToken)
    val (t, t1) = LDA.t(globalTopicCounter, numTopics, beta)
    GlobalParameter(globalTopicCounter, t, t1)
  }

  private def gibbsSampling(): Unit = {
    val broadcast = sc.broadcast(globalParameter)
    val sampleCorpus = sampleToken(corpus, broadcast,
      innerIter + seed, sumToken, numTopics, numTerms, alpha, beta, alphaAS)
    sampleCorpus.persist(storageLevel)

    val counterCorpus = updateCounter(sampleCorpus, numTopics)
    counterCorpus.persist(storageLevel)
    // counterCorpus.vertices.count()
    counterCorpus.edges.count()
    globalParameter = collectGlobalParameter(counterCorpus)

    corpus.edges.unpersist(false)
    corpus.vertices.unpersist(false)
    sampleCorpus.edges.unpersist(false)
    sampleCorpus.vertices.unpersist(false)
    corpus = counterCorpus
    broadcast.unpersist(false)

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
    val totalTopicCounter = this.globalParameter.totalTopicCounter
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

  private[mllib] case class GlobalParameter(totalTopicCounter: BDV[Count],
    t: BDV[Double], t1: BDV[Double])

  private[mllib] case class Parameter(counter: BSV[Count], dist: BSV[Double], dist1: BSV[Double])

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

  private[mllib] def sampleToken(
    graph: Graph[VD, ED],
    broadcast: Broadcast[GlobalParameter],
    innerIter: Long,
    sumTerms: Long,
    numTopics: Int,
    numTerms: Int,
    alpha: Double,
    beta: Double,
    alphaAS: Double): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    graph.mapTriplets(
      (pid, iter) => {
        val gen = new XORShiftRandom(parts * innerIter + pid)
        val wMap = mutable.Map[VertexId, Parameter]()
        val GlobalParameter(totalTopicCounter, t, t1) = broadcast.value
        iter.map {
          triplet =>
            val term = triplet.srcId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr
            val parameter = wMap.getOrElseUpdate(term, w(totalTopicCounter,
              termTopicCounter, numTerms, beta))
            var i = 0
            while (i < topics.length) {
              val currentTopic = topics(i)
              val adjustment = parameter.dist1(currentTopic) + t1(currentTopic)
              val newTopic = metropolisHastingsSampler(gen, parameter.dist, t, adjustment,
                docTopicCounter, termTopicCounter, totalTopicCounter,
                beta, alpha, alphaAS, sumTerms, numTerms, currentTopic)
              assert(newTopic < numTopics)
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
        docTopicCounter = computedModel.generateTopicDistForDocument(docTopicCounter,
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
   * 这里组合使用 Gibbs sampler 和 Metropolis Hastings sampler
   * 每次采样的复杂度为 log(K) K 为主题数(应该可以优化为 (2-6)* log(KD) KD 当前文档包含主题数)
   * 1. 使用 Gibbs sampler 采样标准LDA公式中词相关部分:
   * 论文LightLDA: Big Topic Models on Modest Compute Clusters 公式(6).
   * 2. 把第一步采样得到的概率作为 Proposal q(·) 使用 Metropolis Hastings sampler 采样非对称先验公式
   * 论文 Rethinking LDA: Why Priors Matter 公式(3) .
   */
  def metropolisHastingsSampler(
    rand: Random,
    w: BSV[Double],
    t: BDV[Double],
    adjustment: Double,
    docTopicCounter: VD,
    termTopicCounter: VD,
    totalTopicCounter: BDV[Count],
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numToken: Double,
    numTerms: Double,
    currentTopic: Int): Int = {
    val newTopic = gibbsSamplerWord(rand, w, t, adjustment, currentTopic)
    val ctp = tokenTopicProb(docTopicCounter, termTopicCounter, totalTopicCounter,
      beta, alpha, alphaAS, numToken, numTerms, currentTopic, true)
    val ntp = tokenTopicProb(docTopicCounter, termTopicCounter, totalTopicCounter,
      beta, alpha, alphaAS, numToken, numTerms, newTopic, false)
    val cwp = termTopicProb(termTopicCounter, totalTopicCounter, currentTopic,
      numTerms, beta, true)
    val nwp = termTopicProb(termTopicCounter, totalTopicCounter, newTopic,
      numTerms, beta, false)
    val pi = (ntp * cwp) / (ctp * nwp)

    if (rand.nextDouble() < 0.00001) {
      println(s"Pi: ${pi}")
    }

    if (rand.nextDouble() < math.min(1.0, pi)) {
      newTopic
    } else {
      currentTopic
    }
  }

  @inline private def tokenTopicProb(
    docTopicCounter: VD,
    termTopicCounter: VD,
    totalTopicCounter: BDV[Count],
    beta: Double,
    alpha: Double,
    alphaR: Double,
    numToken: Double,
    numTerms: Double,
    topic: Int,
    isAdjustment: Boolean): Double = {
    val numTopics = docTopicCounter.length
    val adjustment = if (isAdjustment) -1 else 0
    val ratio = (totalTopicCounter(topic) + adjustment + alphaR) / (numToken - 1 + alphaR * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    // 这里移除了常数项 (docLen - 1 + alpha * numTopics)
    (termTopicCounter(topic) + adjustment + beta) * (docTopicCounter(topic) + adjustment + asPrior) /
      (totalTopicCounter(topic) + adjustment + (numTerms * beta))

    // 原始公式论文 Rethinking LDA: Why Priors Matter 公式(3)
    // val docLen = brzSum(docTopicCounter)
    // (termTopicCounter(topic) + adjustment + beta) * (docTopicCounter(topic) + adjustment + asPrior) /
    //   ((totalTopicCounter(topic) + adjustment + (numTerms * beta)) * (docLen - 1 + alpha * numTopics))
  }

  @inline private def termTopicProb(
    termTopicCounter: VD,
    totalTopicCounter: BDV[Count],
    topic: Int,
    numTerms: Double,
    beta: Double,
    isAdjustment: Boolean): Double = {
    val termSum = beta * numTerms
    val count = termTopicCounter(topic)
    if (isAdjustment) {
      (count - 1D + beta) / (totalTopicCounter(topic) - 1D + termSum)
    } else {
      (count + beta) / (totalTopicCounter(topic) + termSum)
    }
  }

  @inline private def indexWord(
    w: BSV[Double],
    t: BDV[Double],
    adjustment: Double,
    currentTopic: Int)(i: Int) = {
    val lastWS = maxMinW(i, w)
    val lastTS = maxMinT(i, t)
    if (i >= currentTopic) {
      lastWS + lastTS + adjustment
    } else {
      lastWS + lastTS
    }
  }

  @inline private def gibbsSamplerWord[V](
    rand: Random,
    w: BSV[Double],
    t: BDV[Double],
    adjustment: Double,
    currentTopic: Int): Int = {
    val numTopics = w.length
    val lastSum = t(numTopics - 1) + w.data(w.used - 1) + adjustment
    val distSum = rand.nextDouble() * lastSum
    if (distSum >= lastSum) {
      return numTopics - 1
    }
    val fun = indexWord(w, t, adjustment, currentTopic) _
    minMaxValueSearch(fun, distSum, numTopics)
  }

  @inline private def w(
    totalTopicCounter: BDV[Count],
    termTopicCounter: VD,
    numTerms: Int,
    beta: Double): Parameter = {
    val numTopics = termTopicCounter.length
    val termSum = beta * numTerms
    val used = termTopicCounter.used
    val index = termTopicCounter.index
    val data = termTopicCounter.data
    val w = new Array[Double](used)
    val w1 = new Array[Double](used)

    var lastWsum = 0D
    var i = 0

    while (i < used) {
      val topic = index(i)
      val count = data(i)
      val lastW = count / (totalTopicCounter(topic) + termSum)
      val lastW1 = (count - 1D) / (totalTopicCounter(topic) - 1D + termSum)
      lastWsum += lastW
      w(i) = lastWsum
      w1(i) = lastW1 - lastW
      i += 1
    }
    Parameter(termTopicCounter, new BSV[Double](index, w, used, numTopics),
      new BSV[Double](index, w1, used, numTopics))
  }

  private def t(
    totalTopicCounter: BDV[Count],
    numTerm: Int,
    beta: Double): (BDV[Double], BDV[Double]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Double](numTopics)
    val t1 = BDV.zeros[Double](numTopics)
    val termSum = beta * numTerm

    var lastTsum = 0D
    for (topic <- 0 until numTopics) {
      val lastT = beta / (totalTopicCounter(topic) + termSum)
      val lastT1 = beta / (totalTopicCounter(topic) - 1.0 + termSum)
      lastTsum += lastT
      t(topic) = lastTsum
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
    assert(index(minMax) <= i)
    if (minMax < array.activeSize - 1) assert(index(minMax + 1) > i)
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

    val topic = if (found) {
      mid
    }
    else if (sum < distSum || isLeft) {
      mid + 1
    } else {
      mid - 1
    }

    assert(index(topic) >= distSum)
    if (topic > 0) assert(index(topic - 1) <= distSum)
    topic
  }

  @inline private[mllib] def maxMinD[V](i: Int, docTopicCounter: BSV[V], d: BDV[Double]) = {
    val lastReturnedPos = minMaxIndexSearch(docTopicCounter, i, -1)
    if (lastReturnedPos > -1) {
      d(docTopicCounter.index(lastReturnedPos))
    }
    else {
      0D
    }
  }

  @inline private[mllib] def maxMinW(i: Int, w: BSV[Double]) = {
    val lastReturnedPos = minMaxIndexSearch(w, i, -1)
    if (lastReturnedPos > -1) {
      w.data(lastReturnedPos)
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
    kryo.register(classOf[LDA.Parameter])
    kryo.register(classOf[LDA.GlobalParameter])

    kryo.register(classOf[Random])
    kryo.register(classOf[LDA])
    kryo.register(classOf[LDAModel])
  }
}
