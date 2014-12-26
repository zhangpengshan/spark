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

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, max => brzMax}

import org.apache.spark.Logging
import org.apache.spark.util.Utils

import Layer._

private[mllib] trait Layer extends Serializable {

  def bias: BDV[Double]

  def weight: BDM[Double]

  def numIn = weight.cols

  def numOut = weight.rows

  def layerType: String

  protected lazy val rand: Random = new Random()

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }

  def forward(input: BDM[Double]): BDM[Double] = {
    assert(input.rows == weight.cols)
    val output: BDM[Double] = weight * input
    for (i <- 0 until output.cols) {
      output(::, i) :+= bias
    }
    computeNeuron(output)
    output
  }

  def backward(input: BDM[Double], delta: BDM[Double]): (BDM[Double], BDV[Double]) = {
    val gradWeight = delta * input.t
    val gradBias = BDV.zeros[Double](numOut)
    for (i <- 0 until input.cols) {
      gradBias :+= delta(::, i)
    }
    (gradWeight, gradBias)
  }

  def computeDeltaTop(
    output: BDM[Double],
    label: BDM[Double]): BDM[Double] = {
    val delta = output - label
    computeNeuronPrimitive(delta, output)
    delta
  }

  def computeDeltaMiddle(output: BDM[Double], nextLayer: Layer,
    nextDelta: BDM[Double]): BDM[Double] = {
    val delta = nextLayer.weight.t * nextDelta
    computeNeuronPrimitive(delta, output)
    delta
  }

  def computeNeuron(temp: BDM[Double]): Unit

  def computeNeuronPrimitive(temp: BDM[Double], output: BDM[Double]): Unit

  protected[neuralNetwork] def sample(out: BDM[Double]): BDM[Double] = out
}

private[mllib] class SigmoidLayer(
  val weight: BDM[Double],
  val bias: BDV[Double]) extends Layer with Logging {

  def this(numIn: Int, numOut: Int) {
    this(initUniformDistWeight(numIn, numOut, 4D * math.sqrt(6D / (numIn + numOut))),
      initializeBias(numOut))
  }

  def layerType: String = "sigmoid"

  def computeNeuron(temp: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (j <- 0 until temp.cols) {
        temp(i, j) = sigmoid(temp(i, j))
      }
    }
  }

  def computeNeuronPrimitive(
    temp: BDM[Double],
    output: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (j <- 0 until temp.cols) {
        temp(i, j) = temp(i, j) * sigmoidPrimitive(output(i, j))
      }
    }
  }

  protected[neuralNetwork] override def sample(input: BDM[Double]): BDM[Double] = {
    input.mapValues(v => if (rand.nextDouble() < v) 1D else 0D)
  }
}


private[mllib] class TanhLayer(
  val weight: BDM[Double],
  val bias: BDV[Double]) extends Layer with Logging {

  def this(numIn: Int, numOut: Int) {
    this(initUniformDistWeight(numIn, numOut, math.sqrt(6D / (numIn + numOut))),
      initializeBias(numOut))
  }

  def layerType: String = "tanh"

  def computeNeuron(temp: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (y <- 0 until temp.cols) {
        temp(i, y) = tanh(temp(i, y))
      }
    }
  }

  def computeNeuronPrimitive(
    temp: BDM[Double],
    output: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (y <- 0 until temp.cols) {
        temp(i, y) = temp(i, y) * tanhPrimitive(output(i, y))
      }
    }
  }

  protected[neuralNetwork] override def sample(input: BDM[Double]): BDM[Double] = {
    input.mapValues(v => if (rand.nextDouble() < v) 1D else 0D)
  }
}

private[mllib] class SoftMaxLayer(
  val weight: BDM[Double],
  val bias: BDV[Double]) extends Layer with Logging {

  def this(numIn: Int, numOut: Int) {
    this(initializeWeight(numIn, numOut), initializeBias(numOut))
  }

  def layerType: String = "softMax"

  def computeNeuron(temp: BDM[Double]): Unit = {
    for (col <- 0 until temp.cols) {
      softMax(temp(::, col))
    }
  }

  def softMax(temp: BDV[Double]): Unit = {
    val max = brzMax(temp)
    var sum = 0D
    for (i <- 0 until temp.length) {
      temp(i) = Math.exp(temp(i) - max)
      sum += temp(i)
    }
    temp :/= sum
  }

  def computeNeuronPrimitive(
    temp: BDM[Double],
    output: BDM[Double]): Unit = {
    // See: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.49.6403

    //  for (i <- 0 until temp.rows) {
    //    for (j <- 0 until temp.cols) {
    //      temp(i, j) = temp(i, j) * softMaxPrimitive(output(i, j))
    //    }
    //  }
  }

  protected[neuralNetwork] override def sample(out: BDM[Double]): BDM[Double] = {
    for (j <- 0 until out.cols) {
      val v = out(::, j)
      var sum = 0D
      var index = 0
      var find = false
      val s = rand.nextDouble()
      while (!find && index < v.length) {
        sum += v(index)
        if (sum >= s) {
          find = true
        } else {
          index += 1
        }
      }
      v :*= 0D
      index = if (find) index else index - 1
      v(index) = 1
    }
    out
  }
}

private[mllib] class NReLuLayer(
  val weight: BDM[Double],
  val bias: BDV[Double]) extends Layer with Logging {
  def this(numIn: Int, numOut: Int) {
    this(initUniformDistWeight(numIn, numOut, 0D, 0.01),
      initializeBias(numOut))
  }

  def layerType: String = "nrelu"

  private def nReLu(tmp: BDM[Double]): Unit = {
    for (i <- 0 until tmp.rows) {
      for (j <- 0 until tmp.cols) {
        val v = tmp(i, j)
        val sd = sigmoid(v)
        val x = v + sd * rand.nextGaussian()
        tmp(i, j) = math.max(0, x)
      }
    }
  }

  def computeNeuron(temp: BDM[Double]): Unit = {
    nReLu(temp)
  }

  def computeNeuronPrimitive(
    temp: BDM[Double],
    output: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (j <- 0 until temp.cols)
        if (output(i, j) <= 0) {
          temp(i, j) = 0
        }
    }
  }
}

private[mllib] class ReLuLayer(
  val weight: BDM[Double],
  val bias: BDV[Double]) extends Layer with Logging {

  def this(numIn: Int, numOut: Int) {
    this(initUniformDistWeight(numIn, numOut, 0.0, 0.01),
      initializeBias(numOut))
  }

  def layerType: String = "relu"

  private def relu(tmp: BDM[Double]): Unit = {
    for (i <- 0 until tmp.rows) {
      for (j <- 0 until tmp.cols) {
        tmp(i, j) = math.max(0, tmp(i, j))
      }
    }
  }

  def computeNeuron(temp: BDM[Double]): Unit = {
    relu(temp)
  }

  def computeNeuronPrimitive(
    temp: BDM[Double],
    output: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (j <- 0 until temp.cols)
        if (output(i, j) <= 0) {
          temp(i, j) = 0
        }
    }
  }

  protected[neuralNetwork] override def sample(input: BDM[Double]): BDM[Double] = {
    input.mapValues { v =>
      val sd = sigmoid(v, 32)
      val x = v + sd * rand.nextGaussian()
      math.max(0, x)
    }
  }
}

private[mllib] class SoftPlusLayer(
  val weight: BDM[Double],
  val bias: BDV[Double]) extends Layer with Logging {
  def this(numIn: Int, numOut: Int) {
    this(initUniformDistWeight(numIn, numOut, 0D, 0.01),
      initializeBias(numOut))
  }

  def layerType: String = "softplus"

  def computeNeuron(temp: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (j <- 0 until temp.cols) {
        temp(i, j) = softplus(temp(i, j))
      }
    }
  }

  def computeNeuronPrimitive(
    temp: BDM[Double],
    output: BDM[Double]): Unit = {
    for (i <- 0 until temp.rows) {
      for (j <- 0 until temp.cols) {
        temp(i, j) *= softplusPrimitive(output(i, j))
      }
    }
  }

  protected[neuralNetwork] override def sample(input: BDM[Double]): BDM[Double] = {
    input.mapValues { v =>
      val sd = sigmoid(v)
      val x = v + sd * rand.nextGaussian()
      // val rng = new NormalDistribution(rand, 0, sd + 1e-23, 1e-9)
      // val x = v + rng.sample()
      math.max(0, x)
    }
  }
}

private[mllib] class GaussianLayer(
  val weight: BDM[Double],
  val bias: BDV[Double]) extends Layer with Logging {

  def this(numIn: Int, numOut: Int) {
    this(initGaussianDistWeight(numIn, numOut), initializeBias(numOut))
  }

  def layerType: String = "gaussian"

  def computeNeuron(tmp: BDM[Double]): Unit = {
    for (i <- 0 until tmp.rows) {
      for (j <- 0 until tmp.cols) {
        val x = tmp(i, j)
        tmp(i, j) = 1 / Math.sqrt(2 * Math.PI) * Math.exp(-x * x / 2)
      }
    }
  }

  def computeNeuronPrimitive(
    temp: BDM[Double],
    output: BDM[Double]): Unit = {
    for (i <- 0 until output.rows) {
      for (j <- 0 until output.cols) {
        val x = output(i, j)
        temp(i, j) *= -x * Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI)
      }
    }
  }

  protected[neuralNetwork] override def sample(input: BDM[Double]): BDM[Double] = {
    input.mapValues(v => v + rand.nextGaussian())
  }
}


private[mllib] object Layer {

  def initializeLayer(
    weight: BDM[Double],
    bias: BDV[Double],
    layerType: String): Layer = {
    layerType match {
      case "gaussian" =>
        new GaussianLayer(weight, bias)
      case "softplus" =>
        new SoftPlusLayer(weight, bias)
      case "relu" =>
        new ReLuLayer(weight, bias)
      case "nrelu" =>
        new NReLuLayer(weight, bias)
      case "softMax" =>
        new SoftMaxLayer(weight, bias)
      case "tanh" =>
        new TanhLayer(weight, bias)
      case "sigmoid" =>
        new SigmoidLayer(weight, bias)
      case _ =>
        throw new IllegalArgumentException("layerType is not correct")
    }
  }

  def initializeBias(numOut: Int): BDV[Double] = {
    BDV.zeros[Double](numOut)
  }

  def initializeWeight(numIn: Int, numOut: Int): BDM[Double] = {
    BDM.zeros[Double](numOut, numIn)
  }

  def initializeWeight(numIn: Int, numOut: Int, rand: () => Double): BDM[Double] = {
    val weight = initializeWeight(numIn, numOut)
    initializeWeight(weight, rand)
  }

  def initializeWeight(w: BDM[Double], rand: () => Double): BDM[Double] = {
    for (i <- 0 until w.data.length) {
      w.data(i) = rand()
    }
    w
  }

  def initUniformDistWeight(numIn: Int, numOut: Int): BDM[Double] = {
    initUniformDistWeight(initializeWeight(numIn, numOut), 0.0)
  }

  def initUniformDistWeight(numIn: Int, numOut: Int, scale: Double): BDM[Double] = {
    initUniformDistWeight(initializeWeight(numIn, numOut), scale)
  }

  def initUniformDistWeight(w: BDM[Double], scale: Double): BDM[Double] = {
    val numIn = w.cols
    val numOut = w.rows
    val s = if (scale <= 0) math.sqrt(6D / (numIn + numOut)) else scale
    initUniformDistWeight(w, -s, s)
  }

  def initUniformDistWeight(numIn: Int, numOut: Int, low: Double, high: Double): BDM[Double] = {
    initUniformDistWeight(initializeWeight(numIn, numOut), low, high)
  }

  def initUniformDistWeight(w: BDM[Double], low: Double, high: Double): BDM[Double] = {
    initializeWeight(w, () => Utils.random.nextDouble() * (high - low) + low)
  }

  def initGaussianDistWeight(numIn: Int, numOut: Int): BDM[Double] = {
    initGaussianDistWeight(initializeWeight(numIn, numOut), 0.0)
  }

  def initGaussianDistWeight(numIn: Int, numOut: Int, scale: Double): BDM[Double] = {
    initGaussianDistWeight(initializeWeight(numIn, numOut), scale)
  }

  def initGaussianDistWeight(weight: BDM[Double], scale: Double): BDM[Double] = {
    val sd = if (scale <= 0) 0.01 else scale
    initializeWeight(weight, () => Utils.random.nextGaussian() * sd)
  }

  @inline def softplus(x: Double, expThreshold: Double = 64): Double = {
    if (x > expThreshold) {
      x
    }
    else if (x < -expThreshold) {
      0
    } else {
      math.log1p(math.exp(x))
    }
  }

  @inline def softplusPrimitive(y: Double, expThreshold: Double = 64): Double = {
    if (y > expThreshold) {
      1
    } else {
      val z = math.exp(y)
      (z - 1) / z
    }

  }

  @inline def tanh(x: Double): Double = {
    val a = math.pow(math.exp(x), 2)
    (a - 1) / (a + 1)
  }

  @inline def tanhPrimitive(y: Double): Double = {
    1 - math.pow(y, 2)
  }

  @inline def sigmoid(x: Double): Double = {
    1d / (1d + math.exp(-x))
  }

  @inline def sigmoid(x: Double, expThreshold: Double): Double = {
    if (x > expThreshold) {
      1D
    } else if (x < -expThreshold) {
      0D
    } else {
      sigmoid(x)
    }
  }

  @inline def sigmoidPrimitive(y: Double): Double = {
    y * (1 - y)
  }

  @inline def softMaxPrimitive(y: Double): Double = {
    y * (1 - y)
  }

  def scalarExp(x: Double, expThreshold: Double = 64D) = {
    if (x < -expThreshold) {
      math.exp(-expThreshold)
    } else if (x > expThreshold) {
      math.exp(-expThreshold)
    }
    else {
      math.exp(x)
    }
  }

  def meanSquaredError(out: BDM[Double], label: BDM[Double]): Double = {
    assert(label.rows == out.rows)
    assert(label.cols == out.cols)
    var diff = 0D
    for (i <- 0 until out.rows) {
      for (j <- 0 until out.cols) {
        diff += math.pow(label(i, j) - out(i, j), 2)
      }
    }
    diff / out.rows
  }

  def crossEntropy(out: BDM[Double], label: BDM[Double]): Double = {
    assert(label.rows == out.rows)
    assert(label.cols == out.cols)
    var cost = 0D
    for (i <- 0 until out.rows) {
      for (j <- 0 until out.cols) {
        val a = label(i, j)
        var b = out(i, j)
        if (b == 0) {
          b += 1e-15
        } else if (b == 1D) {
          b -= 1e-15
        }
        cost += a * math.log(b) + (1 - a) * math.log1p(1 - b)
      }
    }
    (0D - cost) / out.rows
  }
}
