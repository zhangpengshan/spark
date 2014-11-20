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

package org.apache.spark.mllib.neuralNetwork

import java.util.Random

import scala.collection.JavaConversions._

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum => brzSum,
axpy => brzAxpy, norm => brzNorm}
import breeze.numerics.{sigmoid => brzSigmoid}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vector => SV, DenseVector => SDV, Vectors}
import org.apache.spark.mllib.optimization.{Gradient, Updater, LBFGS, GradientDescent}
import org.apache.spark.rdd.RDD

class RBM(
  val weight: BDM[Double],
  val visibleBias: BDV[Double],
  val hiddenBias: BDV[Double])
  extends Logging with Serializable {

  def this(numIn: Int, numOut: Int) {
    this(NNLayer.initializeSigmoidWeight(numIn, numOut),
      NNLayer.initializeSigmoidBias(numOut, numIn),
      NNLayer.initializeSigmoidBias(numIn, numOut))
  }

  private lazy val rand: Random = new Random()

  def cdK: Int = 5

  def numHidden: Int = weight.rows

  def numVisible: Int = weight.cols

  def activateHidden(visible: BDM[Double]): BDM[Double] = {
    assert(visible.rows == weight.cols)
    val hidden: BDM[Double] = weight * visible
    for (i <- 0 until hidden.cols) {
      hidden(::, i) :+= hiddenBias
    }
    brzSigmoid(hidden)
  }

  def activateVisible(hidden: BDM[Double]): BDM[Double] = {
    assert(hidden.rows == weight.rows)
    val visible: BDM[Double] = weight.t * hidden
    for (i <- 0 until visible.cols) {
      visible(::, i) :+= visibleBias
    }
    brzSigmoid(visible)
  }

  def bernoulli(matrix: BDM[Double]): BDM[Double] = {
    matrix.mapValues(v => {
      if (rand.nextDouble() < v) {
        1D
      } else {
        0D
      }
    })
  }

  def learn(visible1: BDM[Double]): (BDM[Double], BDV[Double], BDV[Double], Double, Double) = {
    val batchSize = visible1.cols
    val hidden1 = activateHidden(visible1)
    var visibleK = activateVisible(bernoulli(hidden1))
    var hiddenK = activateHidden(visibleK)

    for (i <- 1 until cdK) {
      visibleK = activateVisible(bernoulli(hiddenK))
      hiddenK = activateHidden(visibleK)
    }

    val gradWeight: BDM[Double] = hidden1 * visible1.t
    gradWeight :-= hiddenK * visibleK.t
    assert(gradWeight.cols == weight.cols)
    assert(gradWeight.rows == weight.rows)
    assert(hidden1.rows == weight.rows)
    assert(visibleK.rows == weight.cols)

    val gV = visible1 - visibleK
    val gradVisibleBias = BDV.zeros[Double](numVisible)
    for (i <- 0 until batchSize) {
      gradVisibleBias :+= gV(::, i)
    }

    val gH = hidden1 - hiddenK
    val gradHiddenBias = BDV.zeros[Double](numHidden)
    for (i <- 0 until batchSize) {
      gradHiddenBias :+= gH(::, i)
    }
    (gradWeight, gradVisibleBias, gradHiddenBias, error(visible1, visibleK), batchSize.toDouble)
  }

  def error(h: BDM[Double], y: BDM[Double]): Double = {
    val error = h - y
    error :*= error
    scala.math.sqrt(brzSum(error) / h.rows)
  }

  def update(
    gradWeight: BDM[Double],
    gradVisibleBias: BDV[Double],
    gradHiddenBias: BDV[Double],
    incWeight: BDM[Double],
    incVisibleBias: BDV[Double],
    incHiddenBias: BDV[Double],
    momentum: Double,
    weightCost: Double,
    learningRate: Double): Double = {
    assert(incVisibleBias.length == weight.cols)
    assert(incHiddenBias.length == weight.rows)
    assert(incWeight.cols == weight.cols)
    assert(incWeight.rows == weight.rows)
    val error = Math.sqrt(gradVisibleBias.dot(gradVisibleBias) / numVisible)

    // gradWeight = (hidden1.t * visible1) - (hiddenK.t * visibleK)
    // gradWeight  -= weights * weightCost
    assert(gradWeight.cols == weight.cols)
    assert(gradWeight.rows == weight.rows)
    brzAxpy(-1 * weightCost, weight, gradWeight)

    // incWeight = incWeight * momentum + gradWeight *
    // learningRate * (1 - momentum)
    incWeight += gradWeight
    gradWeight *= learningRate * (1 - momentum)
    incWeight *= momentum

    // gradVisibleBias = visible1 - visibleK
    assert(gradVisibleBias.size == weight.cols)
    // incVisibleBias = incVisibleBias * momentum + gradVisibleBias * 
    // learningRate * (1 - momentum)
    gradVisibleBias *= (learningRate * (1 - momentum))
    incVisibleBias *= momentum
    incVisibleBias += gradVisibleBias

    // gradHiddenBias = hidden1 - hiddenK
    assert(gradHiddenBias.size == weight.rows)
    // incHiddenBias = incHiddenBias * momentum + gHidden *
    // learningRate * (1 - momentum)
    gradHiddenBias *= (learningRate * (1 - momentum))
    incHiddenBias *= momentum
    incHiddenBias += gradHiddenBias

    weight += incWeight
    visibleBias += incVisibleBias
    hiddenBias += incHiddenBias
    error
  }

  //  def freeEnergy: Double = {
  //    val vh: BDM[Double] = hiddenBias.toDenseMatrix.t * visibleBias.toDenseMatrix
  //    assert(vh.cols == weight.cols)
  //    assert(vh.rows == weight.rows)
  //    vh :*= weight
  //    0 - brzSum(vh)
  //  }

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }
}

object RBM extends Logging {
  def train(
    data: RDD[SV],
    batchSize: Int,
    numIteration: Int,
    numVisible: Int,
    numHidden: Int,
    fraction: Double,
    momentum: Double,
    weightCost: Double,
    learningRate: Double): RBM = {
    train(data, batchSize, numIteration, new RBM(numVisible, numHidden),
      fraction, momentum, weightCost, learningRate)
  }

  def train(
    data: RDD[SV],
    batchSize: Int,
    numIteration: Int,
    rbm: RBM,
    fraction: Double,
    momentum: Double,
    weightCost: Double,
    learningRate: Double): RBM = {
    runSGD(data, rbm, batchSize, numIteration, fraction,
      momentum, weightCost, learningRate)
  }

  def runSGD(
    trainingRDD: RDD[SV],
    batchSize: Int,
    numVisible: Int,
    numHidden: Int,
    maxNumIterations: Int,
    fraction: Double,
    momentum: Double,
    regParam: Double,
    learningRate: Double): RBM = {
    val rbm = new RBM(numVisible, numHidden)
    runSGD(trainingRDD, rbm, batchSize, maxNumIterations, fraction,
      momentum, regParam, learningRate)
  }

  def runSGD(
    data: RDD[SV],
    rbm: RBM,
    batchSize: Int,
    maxNumIterations: Int,
    fraction: Double,
    momentum: Double,
    regParam: Double,
    learningRate: Double): RBM = {
    val numVisible = rbm.numVisible
    val numHidden = rbm.numHidden
    val gradient = new RBMLeastSquaresGradient(numVisible, numHidden)
    val updater = new RBMUpdater(numVisible, numHidden, momentum)
    val optimizer = new GradientDescent(gradient, updater).
      setMiniBatchFraction(fraction).
      setNumIterations(maxNumIterations).
      setRegParam(regParam).
      setStepSize(learningRate)
    val trainingRDD = if (batchSize > 1) {
      batchVector(data, batchSize, numVisible).map(t => (0D, t))
    } else {
      data.map(t => (0D, t))
    }
    val weights = optimizer.optimize(trainingRDD, toVector(rbm))
    fromVector(numVisible, numHidden, weights)
  }

  private[mllib] def batchMatrix(
    data: RDD[SV],
    batchSize: Int,
    numVisible: Int): RDD[BDM[Double]] = {
    val dataBatch = data.sortBy(_.hashCode()).mapPartitions { itr =>
      itr.grouped(batchSize).map { seq =>
        val batch = BDM.zeros[Double](numVisible, seq.size)
        seq.zipWithIndex.foreach { case (v, i) =>
          batch(::, i) :+= v.toBreeze
        }
        batch
      }
    }
    dataBatch
  }

  private[mllib] def batchVector(
    data: RDD[SV],
    batchSize: Int,
    numVisible: Int): RDD[SV] = {
    batchMatrix(data, batchSize, numVisible).map { t =>
      new SDV(t.toArray)
    }
  }

  private[mllib] def fromVector(numVisible: Int, numHidden: Int, weights: SV): RBM = {
    val (weight, visibleBias, hiddenBias) = vectorToStructure(numVisible, numHidden, weights)
    new RBM(weight, visibleBias, hiddenBias)
  }

  private[mllib] def toVector(rbm: RBM): SV = {
    structureToVector(rbm.weight, rbm.visibleBias, rbm.hiddenBias)
  }

  private[mllib] def structureToVector(
    weight: BDM[Double],
    visibleBias: BDV[Double],
    hiddenBias: BDV[Double]): SV = {
    val numVisible = visibleBias.length
    val numHidden = hiddenBias.length
    val sumLen = numHidden * numVisible + numVisible + numHidden
    val data = new Array[Double](sumLen)
    var offset = 0

    System.arraycopy(weight.toArray, 0, data, offset, numHidden * numVisible)
    offset += numHidden * numVisible

    System.arraycopy(visibleBias.toArray, 0, data, offset, numVisible)
    offset += numVisible

    System.arraycopy(hiddenBias.toArray, 0, data, offset, numHidden)
    offset += numHidden

    new SDV(data)
  }

  private[mllib] def vectorToStructure(
    numVisible: Int,
    numHidden: Int,
    weights: SV): (BDM[Double], BDV[Double], BDV[Double]) = {
    val data = weights.toArray
    var offset = 0

    val weight = new BDM[Double](numHidden, numVisible, data, offset)
    offset += numHidden * numVisible

    val visibleBias = new BDV[Double](data, offset, 1, numVisible)
    offset += numVisible

    val hiddenBias = new BDV[Double](data, offset, 1, numHidden)
    offset += numHidden

    (weight, visibleBias, hiddenBias)

  }
}

private[mllib] class RBMLeastSquaresGradient(
  numVisible: Int,
  numHidden: Int,
  batchSize: Int = 1) extends Gradient {

  override def compute(data: SV, label: Double, weights: SV): (SV, Double) = {
    val input = if (data.size > numVisible) {
      val numCol = data.size / numVisible
      new BDM[Double](numVisible, numCol, data.toArray)
    }
    else {
      new BDV(data.toArray, 0, 1, numVisible).toDenseMatrix.t
    }
    val rbm = RBM.fromVector(numVisible, numHidden, weights)
    var (gradWeight, gradVisibleBias, gradHiddenBias, error, numCol) = rbm.learn(input)
    if (numCol != batchSize) {
      val scale: Double = numCol / batchSize
      gradWeight :*= scale
      gradVisibleBias :*= scale
      gradHiddenBias :*= scale
      error *= scale
    }

    (RBM.structureToVector(gradWeight, gradVisibleBias, gradHiddenBias), error)
  }

  override def compute(
    data: SV,
    label: Double,
    weights: SV,
    cumGradient: SV): Double = {
    val (grad, err) = compute(data, label, weights)
    cumGradient.toBreeze += grad.toBreeze
    err
  }
}

private[mllib] class RBMUpdater(
  numVisible: Int,
  numHidden: Int,
  momentum: Double = 0D) extends Updater {

  lazy val incBiasVisible: BDV[Double] = {
    BDV.zeros[Double](numVisible)
  }
  lazy val incBiasHidden: BDV[Double] = {
    BDV.zeros[Double](numHidden)
  }
  lazy val incWeights: BDM[Double] = {
    BDM.zeros[Double](numHidden, numVisible)
  }

  override def compute(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): (SV, Double) = {
    val rbm = RBM.fromVector(numVisible, numHidden, weightsOld)
    val (gradWeight, gradVisibleBias, gradHiddenBias) =
      RBM.vectorToStructure(numVisible, numHidden, gradient)

    if (momentum <= 0D || momentum >= 1D) {
      brzAxpy(-1D * regParam, rbm.weight, gradWeight)
      brzAxpy(stepSize, gradWeight, rbm.weight)
      brzAxpy(stepSize, gradVisibleBias, rbm.visibleBias)
      brzAxpy(stepSize, gradHiddenBias, rbm.hiddenBias)
    }
    else {
      rbm.update(gradWeight, gradVisibleBias, gradHiddenBias,
        incWeights, incBiasVisible, incBiasHidden,
        momentum, regParam, stepSize)
    }

    (RBM.toVector(rbm), regParam)
  }
}
