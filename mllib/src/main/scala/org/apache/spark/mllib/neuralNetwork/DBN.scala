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

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

class DBN(val stackedRBM: StackedRBM, val nn: NN)
  extends Logging with Serializable {
}

object DBN extends Logging {
  def train(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numIteration: Int,
    topology: Array[Int],
    fraction: Double,
    momentum: Double,
    weightCost: Double,
    learningRate: Double): DBN = {
    val dbn = initializeDBN(topology)
    pretrain(data, batchSize, numIteration, dbn,
      fraction, momentum, weightCost, learningRate)
    val newNN = NN.train(data, batchSize, numIteration, dbn.nn,
      fraction, momentum, weightCost, learningRate)
    val oldNN = dbn.nn
    oldNN.innerLayers.zip(newNN.innerLayers).foreach { case (oldLayer, newLayer) =>
      oldLayer.weight := newLayer.weight
      oldLayer.bias := newLayer.bias
    }
    dbn
  }

  private[mllib] def pretrain(
    data: RDD[(SV, SV)],
    batchSize: Int,
    numIteration: Int,
    dbn: DBN,
    fraction: Double,
    momentum: Double,
    weightCost: Double,
    learningRate: Double): DBN = {
    val stackedRBM = dbn.stackedRBM
    val numLayer = stackedRBM.innerRBMs.length
    StackedRBM.train(data.map(_._1), batchSize, numIteration, stackedRBM,
      fraction, momentum, weightCost, learningRate, numLayer - 1)
    dbn
  }

  def initializeDBN(topology: Array[Int]): DBN = {
    val numLayer = topology.length - 1
    val stackedRBM = new StackedRBM(topology)

    val lastRBM = stackedRBM.innerRBMs.last
    NNLayer.initializeSigmoidWeight(lastRBM.weight)
    NNLayer.initializeSigmoidBias(lastRBM.hiddenBias)
    val innerLayers = new Array[NNLayer](numLayer)
    stackedRBM.innerRBMs.zipWithIndex.foreach { case (rbm, index) =>
      innerLayers(index) = new SigmoidLayer(rbm.weight, rbm.hiddenBias)
    }
    val mlp = new NN(innerLayers)

    new DBN(stackedRBM, mlp)
  }
}
