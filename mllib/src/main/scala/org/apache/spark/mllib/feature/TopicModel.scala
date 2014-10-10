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
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Vectors, DenseVector => SDV,
       SparseVector => SSV, Vector => SV}

import TopicModel.Count

class TopicModel private[mllib](
  private[mllib] val _topicCounter: BDV[Count],
  private[mllib] val _topicTermCounter: Array[BSV[Count]],
  private[mllib] val alpha: Double,
  private[mllib] val beta: Double
) extends Serializable {
  def this(topicCounts_ : SDV, topicTermCounts_ : Array[SSV], alpha: Double, beta: Double) {
    this(new BDV[Double](topicCounts_.toArray), topicTermCounts_.map(t =>
      new BSV(t.indices, t.values, t.size)), alpha, beta)
  }

  val (numTopics, numTerms) = (_topicCounter.size, _topicTermCounter.size)

  def topicCounter = Vectors.fromBreeze(_topicCounter)

  def topicTermCounter = _topicTermCounter.map(t => Vectors.fromBreeze(t))

  private[mllib] def update(term: Int, topic: Int, inc: Int) = {
    _topicCounter(topic) += inc
    _topicTermCounter(term)(topic) += inc
    this
  }

  private[mllib] def merge(other: TopicModel) = {
    _topicCounter :+= other._topicCounter
    var i = 0
    while (i < _topicTermCounter.length) {
      _topicTermCounter(i) :+= other._topicTermCounter(i)
      i += 1
    }
    this
  }

  def inference(
    doc: SSV,
    totalIter: Int = 10,
    burnIn: Int = 5,
    rand: Random = new Random()): SV = {
    require(totalIter > burnIn, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnInIter is less than 0")
    infer(doc, totalIter, burnIn, realTime = false, rand)
  }

  /**
   * RT-LDA you can refer to the paper: "Towards Topic Modeling for Big Data",
   * available at [[http://arxiv.org/abs/1405.4402]]
   */
  @Experimental
  def realTimeInference(
    doc: SSV,
    totalIter: Int = 3,
    burnIn: Int = 2,
    rand: Random = new Random()): SV = {
    require(totalIter > burnIn, "totalIter is less than burnIn")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnIn > 0, "burnIn is less than 0")
    infer(doc, totalIter, burnIn, realTime = true, rand)
  }

  private[mllib] def infer(
    doc: SSV,
    totalIter: Int,
    burnIn: Int,
    realTime: Boolean,
    rand: Random): SV = {
    val indices = doc.indices
    val topics = doc.values.map(i => new Array[Int](i.toInt))
    val numTopics = _topicCounter.size
    val topicDist = BSV.zeros[Double](numTopics)
    var docTopicCounter = uniformDistSamplerForDocument(indices, topics, numTopics, rand)

    var i = 0
    while (i < totalIter) {
      docTopicCounter = generateTopicDistForDocument(docTopicCounter,
        indices, topics, realTime, rand)
      if (i + burnIn >= totalIter) topicDist :+= docTopicCounter
      i += 1
    }
    if (burnIn > 1) topicDist :/= (totalIter - burnIn).toDouble
    Vectors.fromBreeze(topicDist)
  }

  private[mllib] def uniformDistSamplerForDocument(
    indices: Array[Int], topics: Array[Array[Int]],
    numTopics: Int, rand: Random): BSV[Count] = {
    val docTopicCounter = BSV.zeros[Count](numTopics)
    var index = 0
    var i = 0
    while (index < indices.length) {
      while (i < topics(index).length) {
        val newTopic = TopicModeling.uniformDistSampler(rand, numTopics)
        topics(index)(i) = newTopic
        docTopicCounter(newTopic) += 1
        i += 1
      }
      i = 0
      index += 1
    }
    docTopicCounter
  }

  private[mllib] def generateTopicDistForDocument(
    docTopicCounter: BSV[Count],
    indices: Array[Int],
    topics: Array[Array[Int]],
    realTime: Boolean,
    rand: Random): BSV[Count] = {
    val newDocTopicCounter = BSV.zeros[Count](numTopics)
    var index = 0
    var i = 0
    while (index < indices.length) {
      val term = indices(index)
      while (i < topics(index).length) {
        val newTopic = if (realTime) {
          max(docTopicCounter, term)
        } else {
          multinomialDistSampler(rand, t, w(term), d(docTopicCounter, term))
        }
        topics(index)(i) = newTopic
        newDocTopicCounter(newTopic) += 1
        i += 1
      }
      i = 0
      index += 1
    }
    newDocTopicCounter
  }

  @inline def maxMinIndexSearch(v: BSV[Double], i: Int, lastReturnedPos: Int): Int = {
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

  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   */
  @inline private def multinomialDistSampler(rand: Random, t: BDV[Double],
    w: BSV[Double], d: BSV[Double]): Int = {
    val numTopics = t.size
    val tSum = t(numTopics - 1)
    val wSum = w(numTopics - 1)
    val dSum = d(numTopics - 1)
    val distSum = rand.nextDouble() * (tSum + wSum + dSum)
    var lastReturnedPos = 0
    var lastWsum = 0D
    var lastDsum = 0D
    var newTopic = numTopics - 1
    if (distSum > tSum + wSum) {
      d.activeIterator.find { case (index, v) =>
        lastReturnedPos = maxMinIndexSearch(w, index, lastReturnedPos)
        if (lastReturnedPos > -1) lastWsum = w.data(lastReturnedPos)
        distSum <= v + lastWsum + t(index)
      }.foreach(t => newTopic = t._1)
    } else if (distSum > tSum) {
      w.activeIterator.find { case (index, v) =>
        lastReturnedPos = maxMinIndexSearch(d, index, lastReturnedPos)
        if (lastReturnedPos > -1) lastDsum = d.data(lastReturnedPos)
        distSum <= lastDsum + v + t(index)
      }.foreach(t => newTopic = t._1)
    } else {
      t.activeIterator.find { case (index, v) =>
        if (w(index) > 0D) lastWsum = w(index)
        if (d(index) > 0D) lastDsum = d(index)
        distSum <= lastDsum + lastWsum + v
      }.foreach(t => newTopic = t._1)
    }

    newTopic
  }

  @inline private def d(docTopicCounter: BV[Count], term: Int): BSV[Double] = {
    val t = BSV.zeros[Double](numTopics)
    var lastSum = 0D
    docTopicCounter.activeIterator.filter(_._2 > 0).foreach { case (i, v) =>
      t(i) = v * (_topicTermCounter(term)(i) + beta) /
        (_topicCounter(i) + (numTerms * beta))
      lastSum = t(i) + lastSum
      t(i) = lastSum
    }
    t(numTopics - 1) = lastSum
    t
  }

  @transient private lazy val t = {
    var i = 0
    val t = BDV.zeros[Double](numTopics)
    var lastSum = 0D
    while (i < numTopics) {
      t(i) = alpha * beta / (_topicCounter(i) + (numTerms * beta))
      lastSum = t(i) + lastSum
      t(i) = lastSum
      i += 1
    }
    t(numTopics - 1) = lastSum
    t
  }

  @transient private lazy val w = {
    val w = new Array[BSV[Double]](numTerms)
    val lastSum = new Array[Double](numTerms)
    for (term <- 0 until numTerms) {
      w(term) = BSV.zeros[Double](numTopics)
      _topicTermCounter(term).activeIterator.foreach { case (topic, cn) =>
        w(term)(topic) = alpha * cn / (_topicCounter(topic) + (numTerms * beta))
        lastSum(term) = w(term)(topic) + lastSum(term)
        w(term)(topic) = lastSum(term)
      }
    }
    for (term <- 0 until numTerms) {
      w(term)(numTopics - 1) = lastSum(term)
    }
    w
  }

  @inline private def max(docTopicCounter: BSV[Count], term: Int): Int = {
    var max = 0D
    var maxIndex = 0
    docTopicCounter.activeIterator.filter(_._2 > 0).foreach { case (k, cn) =>
      val maxK = p(term)(k) * (cn + alpha)
      if (maxK > max) {
        max = maxK
        maxIndex = k
      }
    }
    if (max < r(term)._2) {
      r(term)._1
    }
    else {
      maxIndex
    }
  }

  @transient private lazy val p = {
    _topicTermCounter.map(v => v :/ _topicCounter)
  }

  @transient private lazy val r = {
    p.map { probTerm =>
      var max = 0D
      var i = 0
      probTerm.activeIterator.foreach { case (topic, c) =>
        if (c > max) {
          i = topic
          max = c
        }
      }
      (i, max * alpha)
    }
  }
}

object TopicModel {

  type Count = Double

  def apply(numTopics: Int, numTerms: Int, alpha: Double = 0.1, beta: Double = 0.01) = {
    new TopicModel(
      BDV.zeros[Count](numTopics),
      Array(0 until numTerms: _*).map(_ => BSV.zeros[Count](numTopics)),
      alpha, beta)
  }
}
