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

import PartialConnectedLayer._


trait BaseLayer extends Serializable {
  def layerType: String
}

trait PoolingLayer extends BaseLayer {

  def scaleRows: Int

  def scaleCols: Int

  def forward(input: Array[SM]): Array[SM]

  def computeDelta(
    output: Array[SM],
    nextLayer: PartialConnectedLayer,
    nextDelta: Array[SM]): Array[SM]
}

trait PartialConnectedLayer extends BaseLayer {

  def weight: SM

  def bias: SV

  def connTable: SM

  def inChannels: Int

  def outChannels: Int

  def kernelRows: Int

  def kernelCols: Int

  def forward(input: Array[SM]): Array[SM]

  def backward(input: Array[SM], delta: Array[SM]): (SM, SV)

  def computeDelta(output: Array[SM], nextLayer: PoolingLayer, nextDelta: Array[SM]): Array[SM]

  def computeNeuron(temp: SM): Unit

  def computeNeuronPrimitive(temp: SM, output: SM): Unit

}

class AveragePoolingLayer(
  val scaleRows: Int,
  val scaleCols: Int) extends PoolingLayer {

  override def layerType: String = "average_pooling"

  override def forward(input: Array[SM]): Array[SM] = {
    input.map(m =>
      Matrices.fromBreeze(averageMatrix(m.toBreeze, scaleRows, scaleCols))
    )
  }

  override def computeDelta(
    output: Array[SM],
    nextLayer: PartialConnectedLayer,
    nextDelta: Array[SM]): Array[SM] = {
    val delta = new Array[SM](output.length)
    for (i <- 0 until output.length) {
      var z: BM[Double] = null
      for (j <- 0 until nextDelta.length) {
        val kernel = getKernel(nextLayer.weight, nextLayer.kernelRows, nextLayer.kernelCols, i, j)
        val rk = rot180Matrix(kernel)
        z = convnFull(nextDelta(j).toBreeze, rk, z)
      }
      delta(i) = Matrices.fromBreeze(z)
    }
    delta
  }
}

abstract class ConvolutionalLayer(
  val weight: SM,
  val bias: SV,
  val inChannels: Int,
  val outChannels: Int,
  val kernelRows: Int,
  val kernelCols: Int) extends PartialConnectedLayer {

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

  override def connTable: SM = Matrices.ones(inChannels, outChannels)

  override def forward(input: Array[SM]): Array[SM] = {
    val out = new Array[SM](outChannels)
    for (i <- 0 until outChannels) {
      var z: BM[Double] = null
      for (j <- 0 until inChannels) {
        if (connTable(i, j) > 0D) {
          val kernel = getKernel(weight, kernelRows, kernelCols, j, i)
          z = convnValid(input(j).toBreeze, kernel, z)
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

  override def backward(input: Array[SM], delta: Array[SM]): (SM, SV) = {
    val gradWeight = SDM.zeros(weight.numRows, weight.numCols)
    val gradBias = BDV.zeros[Double](outChannels)
    for (i <- 0 until outChannels) {
      val diff = delta(i).toBreeze
      for (j <- 0 until inChannels) {
        val z = convnValid(input(i).toBreeze, diff)
        val gradKernel = getKernel(gradWeight, kernelRows, kernelCols, j, i)
        gradKernel :+= z
      }
      gradBias(i) = diff.valuesIterator.sum
    }
    (gradWeight, Vectors.fromBreeze(gradBias))
  }

  override def computeDelta(
    output: Array[SM],
    nextLayer: PoolingLayer,
    nextDelta: Array[SM]): Array[SM] = {
    val detla = new Array[SM](outChannels)
    for (i <- 0 until outChannels) {
      detla(i) = Matrices.fromBreeze(expandMatrix(nextDelta(i).toBreeze,
        nextLayer.scaleRows, nextLayer.scaleCols))
      computeNeuronPrimitive(detla(i), output(i))
    }
    detla
  }
}

class CNNReLuLayer(
  weight_ : SM,
  bias_ : SV,
  inChannels_ : Int,
  outChannels_ : Int,
  kernelRows_ : Int,
  kernelCols_ : Int) extends ConvolutionalLayer(
  weight_,
  bias_,
  inChannels_,
  outChannels_,
  kernelRows_,
  kernelCols_) {

  override def layerType: String = "cnn_reLu"

  private def relu(tmp: SM): Unit = {
    for (i <- 0 until tmp.numRows) {
      for (j <- 0 until tmp.numCols) {
        tmp(i, j) = math.max(0, tmp(i, j))
      }
    }
  }

  override def computeNeuron(temp: SM): Unit = {
    relu(temp)
  }

  override def computeNeuronPrimitive(temp: SM, output: SM): Unit = {
    for (i <- 0 until temp.numRows) {
      for (j <- 0 until temp.numCols)
        if (output(i, j) <= 0) {
          temp(i, j) = 0
        }
    }
  }
}

private[neuralNetwork] object PartialConnectedLayer {
  def initializeBias(numOut: Int): SV = {
    new SDV(new Array[Double](numOut))
  }

  def initUniformDistKernel(
    inChannels: Int,
    outChannels: Int,
    kernelRows: Int,
    kernelCols: Int): SM = {
    val w: SM = SDM.zeros(inChannels * kernelRows, outChannels * kernelRows)
    FullyConnectedLayer.initUniformDistWeight(w, 0.01)
    w
  }

  def getKernel(
    weight: SM,
    kernelRows: Int,
    kernelCols: Int,
    inOffset: Int,
    outOffset: Int): BM[Double] = {
    val brzWeight = weight.toBreeze
    val rowStart = inOffset * kernelRows
    val rowEnd = rowStart + kernelRows
    val colStart = outOffset * kernelCols
    val colEnd = colStart + kernelCols
    brzWeight(rowStart until rowEnd, colStart until colEnd)
  }


  def rot180Matrix(matrix: BM[Double]): BM[Double] = {
    val m = matrix.copy
    val mRows = m.rows
    val mCols = m.cols
    for (i <- 0 until mRows) {
      for (j <- 0 until (mCols / 2)) {
        val t = m(i, j)
        m(i, j) = m(i, mCols - 1 - j)
        m(i, mCols - 1 - j) = t
      }
    }

    for (j <- 0 until mCols) {
      for (i <- 0 until (mRows / 2)) {
        val t = m(i, j)
        m(i, j) = m(mRows - 1 - i, j)
        m(mRows - 1 - i, j) = t
      }
    }
    m
  }

  def expandMatrix(matrix: BM[Double], scaleRows: Int, scaleCols: Int): BM[Double] = {
    val mRows = matrix.rows
    val mCols = matrix.cols
    val e = BDM.zeros[Double](mRows * scaleRows, mCols * scaleCols)
    for (si <- 0 until scaleRows) {
      for (sj <- 0 until scaleCols) {
        val mRowsStart = si * mRows
        val mRowsEnd = mRowsStart + mRows
        val mColsStart = sj * mCols
        val mColsEnd = mColsStart + mCols
        e(mRowsStart until mRowsEnd, mColsStart until mColsEnd) := matrix
      }
    }
    e
  }

  def averageMatrix(matrix: BM[Double], scaleRows: Int, scaleCols: Int): BM[Double] = {
    scaleMatrix(matrix, scaleRows, scaleCols, m => m.valuesIterator.sum / (m.rows * m.cols))
  }

  def scaleMatrix(
    matrix: BM[Double],
    scaleRows: Int,
    scaleCols: Int,
    fn: (BM[Double]) => Double): BM[Double] = {
    val mRows = matrix.rows
    val mCols = matrix.cols
    val sRows = mRows / scaleRows
    val sCols = mCols / scaleCols
    val s = BDM.zeros[Double](sRows, sCols)
    for (si <- 0 until sRows) {
      for (sj <- 0 until sCols) {
        val mRowsStart = si * scaleRows
        val mRowsEnd = mRowsStart + scaleRows
        val mColsStart = sj * scaleCols
        val mColsEnd = mColsStart + scaleCols
        s(si, sj) = fn(matrix(mRowsStart until mRowsEnd, mColsStart until mColsEnd))
      }
    }
    s
  }

  def convnFull(matrix: BM[Double], kernel: BM[Double], tmp: BM[Double] = null): BM[Double] = {
    val mRows = matrix.rows
    val mCols = matrix.cols
    val kRows = kernel.rows
    val kCols = kernel.cols
    val eRows = mRows + 2 * (kRows - 1)
    val eCols = mCols + 2 * (kCols - 1)
    val e = BDM.zeros[Double](eRows, eCols)
    for (i <- 0 until mRows) {
      for (j <- 0 until mCols) {
        e(i + kRows - 1, j + kCols - 1) = matrix(i, j)
      }
    }
    convnValid(e, kernel, tmp)
  }

  def convnValid(matrix: BM[Double], kernel: BM[Double], tmp: BM[Double] = null): BM[Double] = {
    val mRows = matrix.rows
    val mCols = matrix.cols
    val kRows = kernel.rows
    val kCols = kernel.cols
    val cRows = mRows - kRows + 1
    val cCols = mCols - kCols + 1
    val out = if (tmp == null) {
      BDM.zeros[Double](cRows, cCols)
    } else {
      tmp
    }

    for (ci <- 0 until cRows) {
      for (cj <- 0 until cCols) {
        var sum = 0.0
        for (ki <- 0 until kRows) {
          for (kj <- 0 until kCols) {
            sum = matrix(ci + ki, cj + kj) * kernel(ki, kj)
          }
        }
        out(ci, cj) += sum
      }
    }
    out
  }
}