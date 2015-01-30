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

import breeze.linalg.{Vector => BV, DenseVector => BDV, SparseVector => BSV, sum => brzSum, norm => brzNorm}
import org.apache.spark.mllib.linalg.{Vectors, DenseVector => SDV, SparseVector => SSV}

class LDAModel private[mllib](
  private[mllib] val gtc: BDV[Double],
  private[mllib] val ttc: Array[BSV[Double]],
  val alpha: Double,
  val beta: Double) extends Serializable {

  def this(topicCounts: SDV, topicTermCounts: Array[SSV], alpha: Double, beta: Double) {
    this(new BDV[Double](topicCounts.toArray), topicTermCounts.map(t =>
      new BSV(t.indices, t.values, t.size)), alpha, beta)
  }

  val (numTopics, numTerms) = (gtc.size, ttc.size)
  lazy val numToken = brzSum(gtc)

  def globalTopicCounter = Vectors.fromBreeze(gtc)

  def topicTermCounter = ttc.map(t => Vectors.fromBreeze(t))

  def inference(doc: SSV, totalIter: Int = 10, burnIn: Int = 5, rand: Random = new Random): SSV = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnInIter is less than 0")

    val topicDist = BSV.zeros[Double](numTopics)
    val bDoc = new BSV[Int](doc.indices, doc.values.map(_.toInt), doc.size)
    val tokens = vec2Array(bDoc)
    val topics = new Array[Int](tokens.length)
    var docTopicCounter = uniformDistSampler(tokens, topics, rand)
    for (i <- 0 until totalIter) {
      docTopicCounter = generateTopicDistForDocument(docTopicCounter, tokens, topics, rand)
      if (i + burnIn >= totalIter) topicDist :+= docTopicCounter
    }
    topicDist.compact()
    topicDist :/= brzNorm(topicDist, 1)
    Vectors.fromBreeze(topicDist).asInstanceOf[SSV]
  }

  private[mllib] def vec2Array(vec: BV[Int]): Array[Int] = {
    val docLen = brzSum(vec)
    var offset = 0
    val sent = new Array[Int](docLen)
    vec.activeIterator.foreach { case (term, cn) =>
      for (i <- 0 until cn) {
        sent(offset) = term
        offset += 1
      }
    }
    sent
  }

  private[mllib] def generateTopicDistForDocument(
    docTopicCounter: BSV[Double],
    tokens: Array[Int],
    topics: Array[Int],
    rand: Random): BSV[Double] = {
    val newDocTopicCounter = BSV.zeros[Double](docTopicCounter.length)
    for (i <- 0 until topics.length) {
      val term = tokens(i)
      val currentTopic = topics(i)
      val newTopic = metropolisHastingsSampler(rand, w(term), t, docTopicCounter, ttc(term), gtc,
        beta, alpha, alpha, numToken, numTerms, currentTopic)
      if (currentTopic != newTopic) {
        newDocTopicCounter(newTopic) += 1D
        // docTopicCounter(currentTopic) -= 1D
      }
    }
    newDocTopicCounter
  }

  private[mllib] def uniformDistSampler(
    tokens: Array[Int],
    topics: Array[Int],
    rand: Random): BSV[Double] = {
    val docTopicCounter = BSV.zeros[Double](numTopics)
    for (i <- 0 until tokens.length) {
      val topic = LDAUtils.uniformDistSampler(rand, numTopics)
      topics(i) = topic
      docTopicCounter(topic) += 1D
    }
    docTopicCounter
  }

  @inline private def gibbsSamplerWord(
    rand: Random,
    t: BDV[Double],
    w: BSV[Double]): Int = {
    val distSum = rand.nextDouble * (t(numTopics - 1) + w.data(w.used - 1))
    val fun = indexWord(t, w) _
    LDAUtils.minMaxValueSearch(fun, distSum, numTopics)
  }

  @inline private def indexWord(
    t: BDV[Double],
    w: BSV[Double])(i: Int) = {
    val lastWS = LDAUtils.maxMinW(i, w)
    val lastTS = LDAUtils.maxMinT(i, t)
    lastWS + lastTS
  }

  // scalastyle:off
  def metropolisHastingsSampler(
    rand: Random,
    w: BSV[Double],
    t: BDV[Double],
    docTopicCounter: BSV[Double],
    termTopicCounter: BSV[Double],
    totalTopicCounter: BDV[Double],
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numToken: Double,
    numTerms: Double,
    currentTopic: Int): Int = {
    val newTopic = gibbsSamplerWord(rand, t, w)
    val ctp = tokenTopicProb(docTopicCounter, termTopicCounter, totalTopicCounter,
      beta, alpha, alphaAS, numToken, numTerms, currentTopic)
    val ntp = tokenTopicProb(docTopicCounter, termTopicCounter, totalTopicCounter,
      beta, alpha, alphaAS, numToken, numTerms, newTopic)
    val cwp = termTopicProb(termTopicCounter, totalTopicCounter, currentTopic, numTerms, beta)
    val nwp = termTopicProb(termTopicCounter, totalTopicCounter, newTopic, numTerms, beta)
    val pi = (ntp * cwp) / (ctp * nwp)

    if (rand.nextDouble() < 0.00001) {
      println(s"Model Pi: ${pi}")
    }

    if (rand.nextDouble() < math.min(1.0, pi)) {
      newTopic
    } else {
      currentTopic
    }
  }
  // scalastyle:on

  @inline private def tokenTopicProb(
    docTopicCounter: BSV[Double],
    termTopicCounter: BSV[Double],
    totalTopicCounter: BDV[Double],
    beta: Double,
    alpha: Double,
    alphaAS: Double,
    numToken: Double,
    numTerms: Double,
    topic: Int): Double = {
    val ratio = (totalTopicCounter(topic) + alphaAS) / (numToken + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    (termTopicCounter(topic) + beta) * (docTopicCounter(topic) + asPrior) /
      (totalTopicCounter(topic) + (numTerms * beta))
  }

  @inline private def termTopicProb(
    termTopicCounter: BSV[Double],
    totalTopicCounter: BDV[Double],
    topic: Int,
    numTerms: Double,
    beta: Double): Double = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  @transient private lazy val t = {
    val t = BDV.zeros[Double](numTopics)
    val termSum = numTerms * beta
    var lastSum = 0D
    for (i <- 0 until numTopics) {
      t(i) = beta / (gtc(i) + termSum)
      lastSum = t(i) + lastSum
      t(i) = lastSum
    }
    t
  }

  @transient private lazy val w = {
    val w = new Array[BSV[Double]](numTerms)
    val termSum = numTerms * beta
    for (term <- 0 until numTerms) {
      val bsv = BSV.zeros[Double](numTopics)
      var lastSum = 0D
      ttc(term).activeIterator.foreach { case (topic, cn) =>
        bsv(topic) = cn / (gtc(topic) + termSum)
        lastSum = bsv(topic) + lastSum
        bsv(topic) = lastSum
      }
      bsv.compact()
      w(term) = bsv
    }
    w
  }

  private[mllib] def merge(term: Int, topic: Int, inc: Int) = {
    gtc(topic) += inc
    ttc(term)(topic) += inc
    this
  }

  private[mllib] def merge(term: Int, counter: BSV[Double]) = {
    ttc(term) :+= counter
    gtc :+= counter
    this
  }

  private[mllib] def merge(other: LDAModel) = {
    gtc :+= other.gtc
    for (i <- 0 until ttc.length) {
      ttc(i) :+= other.ttc(i)
    }
    this
  }

}

object LDAModel {
  def apply(numTopics: Int, numTerms: Int, alpha: Double = 0.1, beta: Double = 0.01) = {
    new LDAModel(
      BDV.zeros[Double](numTopics),
      (0 until numTerms).map(_ => BSV.zeros[Double](numTopics)).toArray, alpha, beta)
  }
}
