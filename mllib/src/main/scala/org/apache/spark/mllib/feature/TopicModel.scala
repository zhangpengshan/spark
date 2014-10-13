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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.{Vectors, DenseVector => SDV, SparseVector => SSV}

import TopicModel.Count

class TopicModel private[mllib](
  private[mllib] val gtc: BDV[Count],
  private[mllib] val ttc: Array[BSV[Count]],
  val alpha: Double,
  val beta: Double) extends Serializable {

  def this(topicCounts: SDV, topicTermCounts: Array[SSV], alpha: Double, beta: Double) {
    this(new BDV[Double](topicCounts.toArray), topicTermCounts.map(t =>
      new BSV(t.indices, t.values, t.size)), alpha, beta)
  }

  val (numTopics, numTerms) = (gtc.size, ttc.size)

  def globalTopicCounter = Vectors.fromBreeze(gtc)

  def topicTermCounter = ttc.map(t => Vectors.fromBreeze(t))

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

  private[mllib] def merge(other: TopicModel) = {
    gtc :+= other.gtc
    for (i <- 0 until ttc.length) {
      ttc(i) :+= other.ttc(i)
    }
    this
  }

  def inference(doc: SSV, totalIter: Int, burnIn: Int, rand: Random): SSV = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnInIter is less than 0")

    val indices = doc.indices
    val topics = doc.values.map(i => new Array[Int](i.toInt))
    val topicDist = BSV.zeros[Double](numTopics)
    var docTopicCounter = uniformDistSamplerForDocument(indices, topics, numTopics, rand)

    var i = 0
    while (i < totalIter) {
      docTopicCounter = generateTopicDistForDocument(docTopicCounter, indices, topics, rand)
      if (i + burnIn >= totalIter) topicDist :+= docTopicCounter
      i += 1
    }
    if (burnIn > 1) topicDist :/= (totalIter - burnIn).toDouble
    Vectors.fromBreeze(topicDist).asInstanceOf[SSV]
  }

  private[mllib] def uniformDistSamplerForDocument(indices: Array[Int], topics: Array[Array[Int]],
    numTopics: Int, rand: Random): BSV[Count] = {
    val docTopicCounter = BSV.zeros[Count](numTopics)
    var index = 0
    while (index < indices.length) {
      var i = 0
      while (i < topics(index).length) {
        val newTopic = TopicModeling.uniformDistSampler(rand, numTopics)
        topics(index)(i) = newTopic
        docTopicCounter(newTopic) += 1
        i += 1
      }
      index += 1
    }
    docTopicCounter
  }

  private[mllib] def generateTopicDistForDocument(
    docTopicCounter: BSV[Count],
    indices: Array[Int],
    topics: Array[Array[Int]],
    rand: Random): BSV[Count] = {
    val newDocTopicCounter = BSV.zeros[Count](numTopics)
    var index = 0
    while (index < indices.length) {
      val term = indices(index)
      var i = 0
      while (i < topics(index).length) {
        val newTopic = multinomialDistSampler(rand, t, w(term), d(docTopicCounter, term))
        topics(index)(i) = newTopic
        newDocTopicCounter(newTopic) += 1
        i += 1
      }
      index += 1
    }
    newDocTopicCounter
  }

  @inline private def maxMinSV(i: Int, w: BSV[Double]) = {
    val lastReturnedPos = TopicModeling.maxMinIndexSearch(w, i, -1)
    if (lastReturnedPos > -1) {
      w.data(lastReturnedPos)
    }
    else {
      0D
    }
  }

  @inline private def index(t: BDV[Double], w: BSV[Double], d: BSV[Double])(i: Int) = {
    val lastDS = maxMinSV(i, d)
    val lastWS = maxMinSV(i, w)
    val lastTS = t(i)
    lastDS + lastWS + lastTS
  }

  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   */
  @inline private def multinomialDistSampler(rand: Random, t: BDV[Double],
    w: BSV[Double], d: BSV[Double]): Int = {
    val numTopics = d.length
    val distSum = rand.nextDouble * (t(numTopics - 1) + w.data(w.used - 1) + d.data(d.used - 1))
    val fun = index(t, w, d) _
    TopicModeling.minMaxValueSearch(fun, distSum, numTopics)
  }

  @inline private def d(docTopicCounter: BV[Count], term: Int): BSV[Double] = {
    val d = BSV.zeros[Double](numTopics)
    var lastSum = 0D
    docTopicCounter.activeIterator.filter(_._2 > 0).foreach { case (topic, cn) =>
      d(topic) = cn * (ttc(term)(topic) + beta) / (gtc(topic) + (numTerms * beta))
      lastSum = d(topic) + lastSum
      d(topic) = lastSum
    }
    d
  }

  @transient private lazy val t = {
    val t = BDV.zeros[Double](numTopics)
    var lastSum = 0D
    for (i <- 0 until numTopics) {
      t(i) = alpha * beta / (gtc(i) + (numTerms * beta))
      lastSum = t(i) + lastSum
      t(i) = lastSum
    }
    t
  }

  @transient private lazy val w = {
    val w = new Array[BSV[Double]](numTerms)
    for (term <- 0 until numTerms) {
      w(term) = BSV.zeros[Double](numTopics)
      var lastSum = 0D
      ttc(term).activeIterator.foreach { case (topic, cn) =>
        w(term)(topic) = alpha * cn / (gtc(topic) + (numTerms * beta))
        lastSum = w(term)(topic) + lastSum
        w(term)(topic) = lastSum
      }
    }
    w
  }
}

object TopicModel {

  type Count = Double

  def apply(numTopics: Int, numTerms: Int, alpha: Double = 0.1, beta: Double = 0.01) = {
    new TopicModel(
      BDV.zeros[Count](numTopics),
      (0 until numTerms).map(_ => BSV.zeros[Count](numTopics)).toArray, alpha, beta)
  }
}
