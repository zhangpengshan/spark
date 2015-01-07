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

private[mllib] trait BaseLayer extends Serializable {
  def layerType: String

  def forward(input: Array[SM]): Array[SM]

  def backward(input: Array[SM], delta: Array[SM]): (SM, SV)

  def outputError(output: Array[SM], label: Array[SM]): Array[SM] = {
    val delta = new Array[SM](output.length)
    for (i <- 0 until output.length) {
      delta(i) = Matrices.fromBreeze(output(i).toBreeze - label(i).toBreeze)
    }
    computeNeuronPrimitive(delta, output)
    delta
  }

  def previousError(
    input: Array[SM],
    previousLayer: BaseLayer,
    currentDelta: Array[SM]): Array[SM]

  def computeNeuron(temp: Array[SM]): Unit

  def computeNeuronPrimitive(temp: Array[SM], output: Array[SM]): Unit
}

private[mllib] trait PoolingLayer extends BaseLayer {
  def scaleRows: Int

  def scaleCols: Int

  def backward(input: Array[SM], delta: Array[SM]): (SM, SV) = {
    null
  }

  def computeNeuron(temp: Array[SM]): Unit = {}

  def computeNeuronPrimitive(temp: Array[SM], output: Array[SM]): Unit = {}
}

private[mllib] trait PartialConnectedLayer extends BaseLayer {
  def weight: SM

  def bias: SV

  def connTable: SM

  def inChannels: Int

  def outChannels: Int

  def kernelRows: Int

  def kernelCols: Int
}

private[mllib] class AveragePoolingLayer(
  val scaleRows: Int,
  val scaleCols: Int) extends PoolingLayer {
  override def layerType: String = "AveragePooling"

  override def forward(input: Array[SM]): Array[SM] = {
    input.map(m =>
      Matrices.fromBreeze(averageMatrix(m.toBreeze, scaleRows, scaleCols))
    )
  }

  override def previousError(
    input: Array[SM],
    previousLayer: BaseLayer,
    currentDelta: Array[SM]): Array[SM] = {
    val preDelta = new Array[SM](input.length)
    for (i <- 0 until input.length) {
      preDelta(i) = Matrices.fromBreeze(expandMatrix(currentDelta(i).toBreeze,
        scaleRows, scaleCols))
    }
    previousLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }
}

private[mllib] trait ConvolutionalLayer extends PartialConnectedLayer {

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
      out(i) = o
    }
    computeNeuron(out)
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

  def previousError(
    input: Array[SM],
    previousLayer: BaseLayer,
    currentDelta: Array[SM]): Array[SM] = {
    val preDelta = new Array[SM](input.length)
    for (i <- 0 until input.length) {
      var z: BM[Double] = null
      for (j <- 0 until currentDelta.length) {
        val kernel = getKernel(weight, kernelRows, kernelCols, i, j)
        val rk = rot180Matrix(kernel)
        z = convnFull(currentDelta(j).toBreeze, rk, z)
      }
      preDelta(i) = Matrices.fromBreeze(z)
    }
    previousLayer.computeNeuronPrimitive(preDelta, input)
    preDelta
  }

}

private[mllib] class ReLuPartialConnectedLayer(
  val weight: SM,
  val bias: SV,
  val inChannels: Int,
  val outChannels: Int,
  val kernelRows: Int,
  val kernelCols: Int) extends ConvolutionalLayer {

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

  override def layerType: String = "cnn_reLu"

  override def computeNeuron(temp: Array[SM]): Unit = {
    temp.foreach(tmp => {
      for (i <- 0 until tmp.numRows) {
        for (j <- 0 until tmp.numCols) {
          tmp(i, j) = math.max(0, tmp(i, j))
        }
      }
    })
  }

  override def computeNeuronPrimitive(temp: Array[SM], output: Array[SM]): Unit = {
    for (index <- 0 until temp.length) {
      val t = temp(index)
      val o = output(index)
      for (i <- 0 until t.numRows) {
        for (j <- 0 until t.numCols)
          if (o(i, j) <= 0) {
            t(i, j) = 0
          }
      }
    }
  }
}

private[mllib] object PartialConnectedLayer {
  def initializeBias(numOut: Int): SV = {
    new SDV(new Array[Double](numOut))
  }

  def initUniformDistKernel(
    inChannels: Int,
    outChannels: Int,
    kernelRows: Int,
    kernelCols: Int): SM = {
    val w: SM = SDM.zeros(inChannels * kernelRows, outChannels * kernelRows)
    NNUtil.initUniformDistWeight(w, 0.01)
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
