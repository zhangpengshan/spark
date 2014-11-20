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

import org.scalatest.{Matchers, FunSuite}
import org.apache.spark.mllib.util.MinstDatasetSuite


class NNSuite extends FunSuite with MinstDatasetSuite with Matchers {

  ignore("NN") {
    val (data, numVisible) = minstTrainDataset(5000)
    data.cache()
    val nn = NN.train(data, 10, 20, Array(numVisible, 10),
      0.1, 0.5, 0.0002, 0.4)
    // NN.runSGD(data, nn, 37, 6000, 0.1, 0.8, 0.0002, 0.1)
    println("Error: " + NN.error(data, nn, 100))
  }

}
