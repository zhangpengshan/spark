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

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, axpy => brzAxpy}
import breeze.numerics.{sigmoid => brzSigmoid}

import org.apache.spark.Logging
import org.apache.spark.util.Utils

private[mllib] trait NNLayer extends Serializable {

  def bias: BDV[Double]

  def weight: BDM[Double]

  def numIn = weight.cols

  def numOut = weight.rows

  protected lazy val rand: Random = new Random()

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }

  def forwardPropagation(input: BDM[Double]): BDM[Double] = {
    assert(input.rows == weight.cols)
    val output: BDM[Double] = weight * input
    for (i <- 0 until output.cols) {
      output(::, i) :+= bias
    }

    computeNeuron(output)
    output
  }

  def backPropagation(input: BDM[Double], delta: BDM[Double]): (BDM[Double], BDV[Double]) = {
    val gradWeight = delta * input.t
    val gradBias = BDV.zeros[Double](numOut)
    for (i <- 0 until input.cols) {
      gradBias :+= delta(::, i)
    }

    (gradWeight, gradBias)
  }

  def computeDeltaTop(output: BDM[Double], label: BDM[Double]): BDM[Double] = {
    val delta = output - label
    computeNeuronDerivative(delta, output)
    delta
  }

  def computeDeltaMiddle(output: BDM[Double], nextLayer: NNLayer,
    nextDelta: BDM[Double]): BDM[Double] = {
    val delta = nextLayer.weight.t * nextDelta
    computeNeuronDerivative(delta, output)
    delta
  }

  def computeNeuron(temp: BDM[Double]): Unit

  def computeNeuronDerivative(temp: BDM[Double], output: BDM[Double]): Unit

}

private[mllib] object NNLayer {

  def initializeSigmoidBias(numIn: Int, numOut: Int): BDV[Double] = {
    val b = BDV.zeros[Double](numOut)
    initializeSigmoidBias(b)
  }

  def initializeSigmoidBias(b: BDV[Double]): BDV[Double] = {
    b
  }

  def initializeSigmoidWeight(numIn: Int, numOut: Int): BDM[Double] = {
    val w = BDM.zeros[Double](numOut, numIn)
    initializeSigmoidWeight(w)
  }

  def initializeSigmoidWeight(w: BDM[Double]): BDM[Double] = {
    val numIn = w.cols
    val numOut = w.rows

    // val scale = 4D * scala.math.sqrt(6D / (numIn + numOut))
    for (i <- 0 until w.data.length) {
      // w.data(i) = (Utils.random.nextDouble() * 2 - 1) * scale
      w.data(i) = (Utils.random.nextDouble() * 4.8D - 2.4D) / (numIn + 1D)
    }
    w
  }
}

private[mllib] class SigmoidLayer(val weight: BDM[Double], val bias: BDV[Double])
  extends NNLayer with Logging {

  def this(numIn_ : Int, numOut_ : Int) {
    this(NNLayer.initializeSigmoidWeight(numIn_, numOut_),
      NNLayer.initializeSigmoidBias(numIn_, numOut_))
  }


  def computeNeuron(temp: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (y <- 0 until temp.cols) {
        temp(i, y) = 1d / (1d + scala.math.exp(0D - temp(i, y)))
      }
    }
  }

  def computeNeuronDerivative(temp: BDM[Double], output: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (y <- 0 until temp.cols) {
        temp(i, y) = temp(i, y) * (output(i, y) * (1 - output(i, y)))
      }
    }
  }
}
