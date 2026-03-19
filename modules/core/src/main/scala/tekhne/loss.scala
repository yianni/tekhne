package tekhne

import tekhne.Linalg._

object Loss:
  def mse(output: Vec, target: Vec): Double =
    require(
      output.length == target.length,
      s"loss dimension mismatch: ${output.length} != ${target.length}"
    )
    output.zip(target).map { case (prediction, expected) =>
      val diff = prediction - expected
      diff * diff
    }.sum / output.length.toDouble

  def mseDerivative(output: Vec, target: Vec): Vec =
    require(
      output.length == target.length,
      s"loss derivative dimension mismatch: ${output.length} != ${target.length}"
    )
    val scaleFactor = 2.0 / output.length.toDouble
    (output - target) * scaleFactor
