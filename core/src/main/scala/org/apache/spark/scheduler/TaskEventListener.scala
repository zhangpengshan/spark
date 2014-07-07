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
package org.apache.spark.scheduler

import org.apache.spark.{ContextCleaner, Logging, SparkConf}

private[spark] class TaskEventListener(appName: String, sparkConf: SparkConf)
  extends SparkListener with Logging {

  val MAX_PROPORTION = 0.7D

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd) {
    val taskMetrics = taskEnd.taskMetrics
    val taskInfo = taskEnd.taskInfo
    val gcProportion = taskMetrics.jvmGCTime.toDouble / taskMetrics.executorRunTime
    if (gcProportion > MAX_PROPORTION) {
      logInfo("task %s:%d on %s gc time was %s exceeds the limit %s,run gc.".format(taskInfo.taskId,
        taskInfo.index, taskInfo.host, (gcProportion * 100).toInt, (MAX_PROPORTION * 100).toInt))
      ContextCleaner.runGC()
    }
  }
}
