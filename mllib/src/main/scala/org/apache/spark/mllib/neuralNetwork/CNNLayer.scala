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

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, Matrix => BM,
max => brzMax, Axis => BrzAxis, sum => brzSum}

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{DenseMatrix => SDM, SparseMatrix => SSM, Matrix => SM,
SparseVector => SSV, DenseVector => SDV, Vector => SV, Vectors, Matrices, BLAS}
import org.apache.spark.util.Utils

import CNNLayer._

trait CNNLayer extends Serializable {

  def weight: SM

  def bias: SV

  def connTable: SM = Matrices.ones(inChannels, outChannels)

  def inChannels: Int

  def outChannels: Int

  def kernelRows: Int

  def kernelCols: Int

  def layerType: String

  protected lazy val rand: Random = new Random()

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }

  def forward(input: Array[SM]): Array[SM]

  def backward(input: SM, delta: SM): (SM, SV)


  def computeDeltaMiddle(output: SM, nextLayer: Layer, nextDelta: SM): SM

  def computeNeuron(temp: SM): Unit

  def computeNeuronPrimitive(temp: SM, output: SM): Unit

}

abstract class ConvolutionalLayer(
  val kernel: Array[Array[SM]],
  val bias: SV,
  val inChannels: Int,
  val outChannels: Int,
  val kernelRows: Int,
  val kernelCols: Int) extends CNNLayer {

  def this(
    inChannels: Int,
    outChannels: Int,
    kernelRows: Int,
    kernelCols: Int) {
    this(initUniformDistKernel(inChannels, outChannels, kernelRows, kernelCols),
      initializeBias(outChannels),
      inChannels,
      outChannels,
      kernelRows,
      kernelCols)
  }

  override def forward(input: Array[SM]): Array[SM] = {
    val brzWeight = weight.toBreeze
    val out = new Array[SM](outChannels)

    for (i <- 0 until outChannels) {
      var z: BM[Double] = null
      for (j <- 0 until inChannels) {
        if (connTable(i, j) > 0D) {
          val rowStart = j * kernelRows
          val rowEnd = rowStart + kernelRows
          val colStart = i * kernelCols
          val colEnd = colStart + kernelCols
          val kernel = brzWeight(rowStart until rowEnd, colStart until colEnd)
          if (z == null) {
            z = convnValid(input(j).toBreeze, kernel)
          } else {
            z :+= convnValid(input(j).toBreeze, kernel)
          }
        }
      }
      assert(z != null)
      z :+= bias(i)
      val o = Matrices.fromBreeze(z)
      computeNeuron(o)
      out(i) = o
    }
    out
  }
}

private[neuralNetwork] object CNNLayer {
  def initializeBias(numOut: Int): SV = {
    new SDV(new Array[Double](numOut))
  }

  def initUniformDistKernel(
    inChannels: Int,
    outChannels: Int,
    kernelRows: Int,
    kernelCols: Int): SM = {
    val w: SM = SDM.zeros(inChannels * kernelRows, outChannels * kernelRows)
    Layer.initUniformDistWeight(w, 0.01)
    w
  }

  def convnValid(matrix: BM[Double], kernel: BM[Double]): BM[Double] = {
    val mRows = matrix.rows
    val mCols = matrix.cols
    val kRows = kernel.rows
    val kCols = kernel.cols
    val cRows = mRows - kRows + 1
    val cCols = mCols - kCols + 1
    val out = BDM.zeros[Double](cRows, cCols)
    for (ci <- 0 until cRows) {
      for (cj <- 0 until cCols) {
        var sum = 0.0
        for (ki <- 0 until kRows) {
          for (kj <- 0 until kCols) {
            sum = matrix(ci + ki, cj + kj) * kernel(ki, kj)
          }
        }
        out(ci, cj) = sum
      }
    }
    out
  }
}