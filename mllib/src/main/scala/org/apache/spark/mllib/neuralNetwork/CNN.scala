package org.apache.spark.mllib.neuralNetwork

import java.util.Random

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM, Matrix => BM,
axpy => brzAxpy, argmax => brzArgMax, max => brzMax, sum => brzSum, norm => brzNorm}

import org.apache.spark.annotation.Experimental
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{DenseMatrix => SDM, SparseMatrix => SSM, Matrix => SM,
SparseVector => SSV, DenseVector => SDV, Vector => SV, Vectors, Matrices, BLAS}
import org.apache.spark.mllib.optimization.{Gradient, Updater, LBFGS, GradientDescent}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD

class CNN(val innerLayers: Array[BaseLayer]) {

}
