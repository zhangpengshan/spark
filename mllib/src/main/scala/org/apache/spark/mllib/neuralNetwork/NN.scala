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

import scala.collection.JavaConversions._

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, axpy => brzAxpy,
sum => brzSum, max => brzMax}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext._
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vector => SV, DenseVector => SDV, Vectors}
import org.apache.spark.mllib.optimization.{Gradient, Updater, LBFGS, GradientDescent}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

class NN(val innerLayers: Array[NNLayer])
  extends Logging with Serializable {
  def this(topology_ : Array[Int]) {
    this(NN.initializeLayers(topology_))
  }

  lazy val topology: Array[Int] = {
    val topology = new Array[Int](numLayer + 1)
    topology(0) = numInput
    for (i <- 1 to numLayer) {
      topology(i) = innerLayers(i - 1).numOut
    }
    topology
  }

  def numLayer = innerLayers.length

  def numInput = innerLayers.head.numIn

  def numOut = innerLayers.last.numOut

  def predict(x: BDM[Double]): BDM[Double] = {
    var output = x
    for (layer <- 0 until numLayer) {
      output = innerLayers(layer).forwardPropagation(output)
    }
    output
  }

  def learn(x: BDM[Double], label: BDM[Double]): (Array[(BDM[Double], BDV[Double])],
    Double, Double) = {
    val in = new Array[BDM[Double]](numLayer)
    val out = new Array[BDM[Double]](numLayer)
    val delta = new Array[BDM[Double]](numLayer)
    val grads = new Array[(BDM[Double], BDV[Double])](numLayer)

    for (layer <- 0 until numLayer) {
      val input = if (layer == 0) {
        x
      } else {
        out(layer - 1)
      }
      in(layer) = input

      val output = innerLayers(layer).forwardPropagation(input)
      out(layer) = output
    }

    for (layer <- (0 until numLayer).reverse) {
      val input = in(layer)
      val output = out(layer)
      delta(layer) = if (layer == numLayer - 1) {
        innerLayers(layer).computeDeltaTop(output, label)
      } else {
        innerLayers(layer).computeDeltaMiddle(output, innerLayers(layer + 1), delta(layer + 1))
      }
      grads(layer) = innerLayers(layer).backPropagation(input, delta(layer))
    }

    (grads, error(out.last, label), x.cols.toDouble)
  }

  def update(
    gradWeights: Array[BDM[Double]],
    gradBias: Array[BDV[Double]],
    incWeights: Array[BDM[Double]],
    incBias: Array[BDV[Double]],
    momentum: Double,
    weightCost: Double,
    learningRate: Double
    ): Double = {
    var error = 0D
    for (layer <- (0 until numLayer).reverse) {
      error += scala.math.sqrt(gradBias(layer).dot(gradBias(layer)) / gradBias(layer).length)

      // gradWeights(layer)  -= incWeights(layer) * weightCost
      brzAxpy(-1 * weightCost, innerLayers(layer).weight, gradWeights(layer))

      // incWeights = incWeights * momentum + incWeights * learningRate * (1 - momentum)
      gradWeights(layer) *= learningRate * (1 - momentum)
      incWeights(layer) *= momentum
      incWeights(layer) += gradWeights(layer)

      error += scala.math.sqrt(gradBias(layer).dot(gradBias(layer)) / gradBias(layer).length)
      // incBias = incBias * momentum + gradBias * learningRate * (1 - momentum)
      gradBias(layer) *= (learningRate * (1 - momentum))
      incBias(layer) *= momentum
      incBias(layer) += gradBias(layer)

      innerLayers(layer).weight -= incWeights(layer)
      innerLayers(layer).bias -= incBias(layer)
    }
    error
  }

  def error(h: BDM[Double], y: BDM[Double]): Double = {
    val error = h - y
    error :*= error
    brzSum(error) * 0.5
  }
}

object NN {

  def train(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numIteration: Int,
    topology: Array[Int],
    fraction: Double,
    momentum: Double,
    weightCost: Double,
    learningRate: Double): NN = {
    train(data, batchSize, numIteration, new NN(topology),
      fraction, momentum, weightCost, learningRate)
  }

  def train(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numIteration: Int,
    nn: NN,
    fraction: Double,
    momentum: Double,
    weightCost: Double,
    learningRate: Double): NN = {
    runLBFGS(data, nn, batchSize, numIteration, Double.MinPositiveValue, momentum, weightCost)
  }

  def runSGD(
    trainingRDD: RDD[(SV, SV)],
    topology: Array[Int],
    batchSize: Int,
    maxNumIterations: Int,
    fraction: Double,
    momentum: Double,
    regParam: Double,
    learningRate: Double): NN = {
    val nn = new NN(topology)
    runSGD(trainingRDD, nn, batchSize, maxNumIterations, fraction,
      momentum, regParam, learningRate)
  }

  def runSGD(
    data: RDD[(SV, SV)],
    nn: NN,
    batchSize: Int,
    maxNumIterations: Int,
    fraction: Double,
    momentum: Double,
    regParam: Double,
    learningRate: Double
    ): NN = {
    val topology: Array[Int] = nn.topology
    val gradient = new NNLeastSquaresGradient(topology)
    val updater = new NNUpdater(topology, momentum)
    val optimizer = new GradientDescent(gradient, updater).
      setMiniBatchFraction(fraction).
      setNumIterations(maxNumIterations).
      setRegParam(regParam).
      setStepSize(learningRate)
    val trainingRDD: RDD[(Double, SV)] = batchVector(data, batchSize,
      nn.numInput, nn.numOut).map(t => (0D, t))
    val weights = optimizer.optimize(trainingRDD, toVector(nn))
    fromVector(topology, weights)
  }

  def runLBFGS(
    trainingRDD: RDD[(SV, SV)],
    topology: Array[Int],
    batchSize: Int,
    maxNumIterations: Int,
    convergenceTol: Double,
    momentum: Double,
    regParam: Double): NN = {
    val nn = new NN(topology)
    runLBFGS(trainingRDD, nn, batchSize, maxNumIterations, convergenceTol, momentum, regParam)
  }

  def runLBFGS(
    data: RDD[(SV, SV)],
    nn: NN,
    batchSize: Int,
    maxNumIterations: Int,
    convergenceTol: Double,
    momentum: Double,
    regParam: Double): NN = {
    val topology: Array[Int] = nn.topology
    val gradient = new NNLeastSquaresGradient(topology, batchSize)
    val updater = new NNUpdater(topology, momentum)
    val optimizer = new LBFGS(gradient, updater).
      setConvergenceTol(convergenceTol).
      setNumIterations(maxNumIterations).
      setRegParam(regParam)

    val trainingRDD: RDD[(Double, SV)] = if (batchSize > 1) {
      batchVector(data, batchSize,
        nn.numInput, nn.numOut).map(t => (0D, t))
    }
    else {
      data.map(v =>
        (0.0,
          Vectors.fromBreeze(BDV.vertcat(
            v._1.toBreeze.toDenseVector,
            v._2.toBreeze.toDenseVector))
          ))
    }

    val weights = optimizer.optimize(trainingRDD, toVector(nn))
    fromVector(topology, weights)
  }

  private[mllib] def fromVector(topology: Array[Int], weights: SV): NN = {
    val layers: Array[NNLayer] = vectorToStructure(topology, weights).map { case (weight, bias) =>
      new SigmoidLayer(weight, bias)
    }
    new NN(layers)
  }

  private[mllib] def toVector(nn: NN): SV = {
    structureToVector(nn.innerLayers.map(l => (l.weight, l.bias)))
  }

  private[mllib] def structureToVector(grads: Array[(BDM[Double], BDV[Double])]): SV = {
    val numLayer = grads.length
    val sumLen = grads.map(m => m._1.rows * m._1.cols + m._2.length).sum
    val data = new Array[Double](sumLen)
    var offset = 0
    for (l <- 0 until numLayer) {
      val (gradWeight, gradBias) = grads(l)
      val numIn = gradWeight.cols
      val numOut = gradWeight.rows
      System.arraycopy(gradWeight.toArray, 0, data, offset, numOut * numIn)
      offset += numIn * numOut
      System.arraycopy(gradBias.toArray, 0, data, offset, numOut)
      offset += numOut
    }
    new SDV(data)
  }

  private[mllib] def vectorToStructure(
    topology: Array[Int],
    weights: SV): Array[(BDM[Double], BDV[Double])] = {
    val data = weights.toArray
    val numLayer = topology.length - 1
    val grads = new Array[(BDM[Double], BDV[Double])](numLayer)
    var offset = 0
    for (layer <- 0 until numLayer) {
      val numIn = topology(layer)
      val numOut = topology(layer + 1)
      val weight = new BDM[Double](numOut, numIn, data, offset)
      offset += numIn * numOut
      val bias = new BDV[Double](data, offset, 1, numOut)
      offset += numOut
      grads(layer) = (weight, bias)
    }
    grads
  }

  def error(data: RDD[(SV, SV)], nn: NN, batchSize: Int): Double = {
    val count = data.count()
    val dataBatches = batchMatrix(data, batchSize, nn.numInput, nn.numOut)
    val sumError = dataBatches.map { case (x, y) =>
      val h = nn.predict(x)
      (0 until h.cols).map(i => {
        val max = brzMax(h(::, i))
        val b = h(::, i).mapValues(d => {
          if (max == d) {
            1D
          } else {
            0D
          }
        })

        if (b == y(::, i)) {
          0D
        } else {
          1D
        }
      }).sum
    }.sum
    sumError / count
  }

  private[mllib] def batchMatrix(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numInput: Int,
    numOut: Int): RDD[(BDM[Double], BDM[Double])] = {
    val dataBatch = data.mapPartitions { itr =>
      itr.grouped(batchSize).map { seq =>
        val x = BDM.zeros[Double](numInput, seq.size)
        val y = BDM.zeros[Double](numOut, seq.size)
        seq.zipWithIndex.foreach { case (v, i) =>
          x(::, i) :+= v._1.toBreeze
          y(::, i) :+= v._2.toBreeze
        }
        (x, y)
      }
    }
    dataBatch
  }

  private[mllib] def batchVector(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numInput: Int,
    numOut: Int): RDD[SV] = {
    batchMatrix(data, batchSize, numInput, numOut).map { t =>
      val input = t._1
      val label = t._2
      val sumLen = (input.rows + label.rows) * input.cols
      val data = new Array[Double](sumLen)
      var offset = 0
      System.arraycopy(input.toArray, 0, data, offset, input.rows * input.cols)
      offset += input.rows * input.cols

      System.arraycopy(label.toArray, 0, data, offset, label.rows * input.cols)
      offset += input.rows * input.cols
      new SDV(data)
    }
  }

  private[mllib] def initializeLayers(topology: Array[Int]): Array[NNLayer] = {
    val numLayer = topology.length - 1
    val layers = new Array[NNLayer](numLayer)
    var nextLayer: NNLayer = null
    for (layer <- (0 until numLayer).reverse) {
      layers(layer) = new SigmoidLayer(topology(layer), topology(layer + 1))
      nextLayer = layers(layer)
      println(s"layers($layer) = ${layers(layer).numIn} * ${layers(layer).numOut}")
    }
    layers
  }

}

private[mllib] class NNLeastSquaresGradient(
  topology: Array[Int],
  batchSize: Int = 1) extends Gradient {

  override def compute(data: SV, label: Double, weights: SV): (SV, Double) = {
    val numIn = topology.head
    val numLabel = topology.last
    var input: BDM[Double] = null
    var label: BDM[Double] = null
    val batchedData = data.toArray
    if (data.size != numIn + numLabel) {
      val numCol = data.size / (numIn + numLabel)
      input = new BDM[Double](numIn, numCol, batchedData)
      label = new BDM[Double](numLabel, numCol, batchedData, numIn * numCol)
    }
    else {
      input = new BDV(batchedData, 0, 1, numIn).toDenseMatrix.t
      label = new BDV(batchedData, numIn, 1, numLabel).toDenseMatrix.t
    }

    val nn = NN.fromVector(topology, weights)
    var (grads, error, numCol) = nn.learn(input, label)
    if (numCol != batchSize) {
      val scale: Double = numCol / batchSize
      grads.foreach { t =>
        t._1 :*= scale
        t._2 :*= scale
      }
      error *= scale
    }

    (NN.structureToVector(grads), error)
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

private[mllib] class NNUpdater(val topology: Array[Int], momentum: Double = 0D) extends Updater {

  def numLayer = topology.length - 1

  lazy val incBias: Array[BDV[Double]] = {
    val incBias = new Array[BDV[Double]](numLayer)
    for (layer <- 0 until numLayer) {
      val numRow = topology(layer + 1)
      val numCol = topology(layer)
      incBias(layer) = BDV.zeros[Double](numRow)
    }
    incBias
  }

  lazy val incWeights: Array[BDM[Double]] = {
    val incWeights = new Array[BDM[Double]](numLayer)
    for (layer <- 0 until numLayer) {
      val numRow = topology(layer + 1)
      val numCol = topology(layer)
      incWeights(layer) = BDM.zeros[Double](numRow, numCol)
    }
    incWeights
  }

  override def compute(
    weightsOld: SV,
    gradient: SV,
    stepSize: Double,
    iter: Int,
    regParam: Double): (SV, Double) = {
    val nn = NN.fromVector(topology, weightsOld)
    val grads = NN.vectorToStructure(topology, gradient)

    //    if (momentum <= 0D || momentum >= 1D) {
    //      val innerLayers = nn.innerLayers
    //      for (layer <- 0 until innerLayers.length) {
    //        brzAxpy(-1D * regParam, innerLayers(layer).weight, grads(layer)._1)
    //        brzAxpy(-1D * stepSize, grads(layer)._1, innerLayers(layer).weight)
    //        brzAxpy(-1D * stepSize, grads(layer)._2, innerLayers(layer).bias)
    //      }
    //    } else {
    val (gradWeighs, gradBias) = grads.unzip
    nn.update(gradWeighs.toArray, gradBias.toArray, incWeights, incBias,
      momentum, regParam, stepSize)
    //    }

    (NN.toVector(nn), regParam)
  }
}
