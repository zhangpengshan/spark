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
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

case class Document(docId: Int, content: Iterable[Int], var topics: Iterable[Int] = null,
                    var topicDist: BV[Double] = null)

class TopicModel(val topicCounts_ : BDV[Double],
                 val topicTermCounts_ : Array[BSV[Double]],
                 val alpha: Double,
                 val beta: Double)
  extends Serializable {

  def this(topicCounts_ : SDV, topicTermCounts_ : Array[SSV], alpha: Double, beta: Double) =
    this(new BDV[Double](topicCounts_.toArray), topicTermCounts_.map(t =>
      new BSV(t.indices, t.values, t.size)), alpha, beta)

  def topicCounts = Vectors.dense(topicCounts_.toArray)

  def topicTermCounts = topicTermCounts_.map(t => Vectors.sparse(t.size, t.activeIterator.toSeq))

  def update(term: Int, topic: Int, inc: Int) = {
    topicCounts_(topic) += inc
    topicTermCounts_(topic)(term) += inc
    this
  }

  def merge(other: TopicModel) = {
    topicCounts_ += other.topicCounts_
    var i = 0
    while (i < topicTermCounts_.length) {
      topicTermCounts_(i) += other.topicTermCounts_(i)
      i += 1
    }
    this
  }

  def phi(topic: Int, term: Int): Double = {
    val numTerms = topicTermCounts_.head.size
    (topicTermCounts_(topic)(term) + beta) / (topicCounts_(topic) + numTerms * beta)
  }

  def infer(doc: Document, rand: Random, totalIter: Int = 10, burnInIter: Int = 5): BV[Double] = {
    require(totalIter > burnInIter, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnInIter > 0, "burnInIter is less than 0")

    val Document(_, content, _, topicDist) = doc
    val numTopics = topicCounts_.size
    var lastTopicDist = BSV.zeros[Double](numTopics)
    var currentTopicDist = topicDist
    val probDist = BSV.zeros[Double](numTopics)

    for (i <- 1 to totalIter) {
      if (currentTopicDist != null) {
        content.foreach { term =>
          val dist = generateTopicDistForTerm(currentTopicDist, term)
          val lastTopic = CGS.multinomialDistSampler(rand, dist)
          lastTopicDist(lastTopic) += 1
        }
      }
      else {
        content.foreach { term =>
          val lastTopic = CGS.uniformDistSampler(rand, numTopics)
          lastTopicDist(lastTopic) += 1
        }
      }

      if (i > burnInIter) {
        probDist :+= lastTopicDist
      }
      currentTopicDist = lastTopicDist
      lastTopicDist = BSV.zeros[Double](numTopics)
    }

    probDist :/= (totalIter - burnInIter).toDouble
    probDist
  }


  /**
   * This function used for computing the new distribution after drop one from current document,
   * which is a really essential part of Gibbs sampling for LDA, you can refer to the paper:
   * <I>Parameter estimation for text analysis<I>
   */
  def dropOneDistSampler(docTopicCount: BV[Double], rand: Random, term: Int,
                         currentTopic: Int): Int = {
    val topicThisTerm = generateTopicDistForTerm(docTopicCount, term,
      currentTopic, isTrainModel = true)
    CGS.multinomialDistSampler(rand, topicThisTerm)
  }

  def generateTopicDistForTerm(docTopicCount: BV[Double], term: Int,
                               currentTopic: Int = -1, isTrainModel: Boolean = false):
  BDV[Double] = {
    val (numTopics, numTerms) = (topicCounts_.size, topicTermCounts_.head.size)
    val topicThisTerm = BDV.zeros[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      val adjustment = if (isTrainModel && i == currentTopic) -1 else 0
      topicThisTerm(i) = (topicTermCounts_(i)(term) + adjustment + beta) /
        (topicCounts_(i) + adjustment + (numTerms * beta)) *
        (docTopicCount(i) + adjustment + alpha)
      i += 1
    }
    topicThisTerm
  }
}

object TopicModel {
  def apply(numTopics: Int, numTerms: Int, alpha: Double = 0.1,
            beta: Double = 0.01) = new TopicModel(
    BDV.zeros[Double](numTopics),
    Array(0 until numTopics: _*).map(_ => BSV.zeros[Double](numTerms)),
    alpha, beta)

}

class LDA private(
                   var numTopics: Int,
                   var numTerms: Int,
                   totalIter: Int,
                   burnInIter: Int,
                   var alpha: Double,
                   var beta: Double)
  extends Serializable with Logging {
  def run(input: RDD[Document]): (TopicModel, RDD[Document]) = {
    val initModel = TopicModel(numTopics, numTerms, alpha, beta)
    CGS.runGibbsSampling(input, initModel, totalIter, burnInIter)
  }
}

object LDA extends Logging {
  def train(
             data: RDD[Document],
             numTerms: Int,
             numTopics: Int,
             totalIter: Int,
             burnInIter: Int,
             alpha: Double,
             beta: Double):
  (TopicModel, RDD[Document]) = {
    val lda = new LDA(numTopics, numTerms, totalIter, burnInIter, alpha, beta)
    lda.run(data)
  }

  /**
   * Perplexity is a kind of evaluation method of LDA. Usually it is used on unseen data. But here
   * we use it for current documents, which is also OK. If using it on unseen data, you must do an
   * iteration of Gibbs sampling before calling this. Small perplexity means good result.
   */
  def perplexity(data: RDD[Document], computedModel: TopicModel): Double = {
    val broadcastModel = data.context.broadcast(computedModel)
    val (termProb, totalNum) = data.mapPartitions { docs =>
      val model = broadcastModel.value
      val numTopics = model.topicCounts_.size
      val numTerms = model.topicTermCounts_.head.size
      val rand = new Random
      val alpha = model.alpha
      docs.flatMap { case (doc@Document(docId, content, topics, topicDist)) =>
        val currentTheta = BSV.zeros[Double](numTerms)
        val theta = model.infer(doc, rand)
        content.foreach { term =>
          (0 until numTopics).foreach { topic =>
            currentTheta(term) += model.phi(topic, term) * ((theta(topic) + alpha) /
              (content.size + alpha * numTopics))
          }
        }
        content.map(x => (math.log(currentTheta(x)), 1))
      }
    }.reduce { (lhs, rhs) =>
      (lhs._1 + rhs._1, lhs._2 + rhs._2)
    }
    math.exp(-1 * termProb / totalNum)
  }

}

/**
 * Collapsed Gibbs sampling from for Latent Dirichlet Allocation.
 */
object CGS extends Logging {

  /**
   * Main function of running a Gibbs sampling method. It contains two phases of total Gibbs
   * sampling: first is initialization, second is real sampling.
   */
  def runGibbsSampling(data: RDD[Document], initModel: TopicModel,
                       totalIter: Int, burnInIter: Int): (TopicModel, RDD[Document]) = {
    require(totalIter > burnInIter, "totalIter is less than burnInIter")
    require(totalIter > 0, "totalIter is less than 0")
    require(burnInIter > 0, "burnInIter is less than 0")

    val (numTopics, numTerms, alpha, beta) = (initModel.topicCounts_.size,
      initModel.topicTermCounts_.head.size,
      initModel.alpha, initModel.beta)
    val probModel = TopicModel(numTopics, numTerms, alpha, beta)

    // Construct topic assignment RDD
    logInfo("Start initialization")
    var (params, docTopics) = sampleTermAssignment(data, initModel)

    for (iter <- 1 to totalIter) {
      logInfo("Start Gibbs sampling (Iteration %d/%d)".format(iter, totalIter))
      val broadcastParams = data.context.broadcast(params)
      val previousDocTopics = docTopics
      docTopics = docTopics.mapPartitions { docs =>
        val rand = new Random
        val currentParams = broadcastParams.value
        docs.map { case Document(docId, content, topics, topicDist) =>
          val chosenTopicCounts: BV[Double] = BSV.zeros[Double](numTopics)
          val chosenTopics = content.zip(topics).map { case (term, topic) =>
            val chosenTopic = currentParams.dropOneDistSampler(topicDist, rand, term, topic)
            if (topic != chosenTopic) {
              topicDist(topic) += -1
              currentParams.update(term, topic, -1)
              currentParams.update(term, chosenTopic, 1)
              topicDist(chosenTopic) += 1
            }
            chosenTopicCounts(chosenTopic) += 1
            chosenTopic
          }
          Document(docId, content, chosenTopics, chosenTopicCounts)
        }
      }.setName(s"LDA-$iter").cache()

      if (iter % 20 == 0 && data.context.getCheckpointDir.isDefined) {
        docTopics.checkpoint()
      }

      params = collectTopicCounters(docTopics, numTerms, numTopics)

      if (iter > burnInIter) {
        probModel.merge(params)
      }
      previousDocTopics.unpersist()
    }
    val burnIn = (totalIter - burnInIter).toDouble
    probModel.topicCounts_ :/= burnIn
    probModel.topicTermCounts_.foreach(_ :/= burnIn)
    (probModel, docTopics)
  }

  private def collectTopicCounters(docTopics: RDD[Document], numTerms: Int, numTopics: Int):
  TopicModel = {
    docTopics.mapPartitions { iter =>
      val topicCounters = TopicModel(numTopics, numTerms)
      iter.foreach { doc =>
        doc.content.zip(doc.topics).foreach(t => topicCounters.update(t._1, t._2, 1))
      }
      Iterator(topicCounters)
    }.fold(TopicModel(numTopics, numTerms)) { (thatOne, otherOne) =>
      thatOne.merge(otherOne)
    }
  }

  /**
   * Initial step of Gibbs sampling, which supports incremental LDA.
   */
  private def sampleTermAssignment(data: RDD[Document], topicModel: TopicModel):
  (TopicModel, RDD[Document]) = {
    val (numTopics, numTerms, alpha, beta) = (topicModel.topicCounts_.size,
      topicModel.topicTermCounts_.head.size,
      topicModel.alpha, topicModel.beta)
    val broadcastParams = data.context.broadcast(topicModel)

    val initialDocs = if (topicModel.topicCounts_.norm(2) == 0) {
      data.mapPartitions { docs =>
        val rand = new Random
        docs.map { case Document(docId, content, topics, topicDist) =>
          val lastDocTopicCount = BSV.zeros[Double](numTopics)
          val lastTopics = content.map { term =>
            val topic = uniformDistSampler(rand, numTopics)
            lastDocTopicCount(topic) += 1
            topic
          }
          Document(docId, content, lastTopics, lastDocTopicCount)
        }
      }.cache()
    } else {
      data.mapPartitions { docs =>
        val rand = new Random
        val currentParams = broadcastParams.value
        docs.map { case Document(docId, content, topics, topicDist) =>
          val lastDocTopicCount = BSV.zeros[Double](numTopics)
          val lastTopics = content.map { term =>
            val dist = currentParams.generateTopicDistForTerm(topicDist, term)
            val lastTopic = multinomialDistSampler(rand, dist)
            lastDocTopicCount(lastTopic) += 1
            lastTopic
          }
          Document(docId, content, lastTopics, lastDocTopicCount)
        }
      }.cache()
    }

    val initialModel = collectTopicCounters(initialDocs, numTerms, numTopics)
    (initialModel, initialDocs)
  }

  /**
   * A uniform distribution sampler, which is only used for initialization.
   */
  def uniformDistSampler(rand: Random, dimension: Int): Int = rand.nextInt(dimension)

  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   */
  def multinomialDistSampler(rand: Random, dist: BDV[Double]): Int = {
    val distSum = rand.nextDouble() * breeze.linalg.sum[BDV[Double], Double](dist)

    def loop(index: Int, accum: Double): Int = {
      if (index == dist.length) return dist.length - 1
      val sum = accum - dist(index)
      if (sum <= 0) index else loop(index + 1, sum)
    }

    loop(0, distSum)
  }
}
