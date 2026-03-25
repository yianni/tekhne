package tekhne

import tekhne.Linalg._

/** Loss functions used during training. */
object Loss:
  /** Mean squared error averaged across output dimensions. */
  def mse(output: Vec, target: Vec): Double =
    require(
      output.length == target.length,
      s"loss dimension mismatch: ${output.length} != ${target.length}"
    )
    output.zip(target).map { case (prediction, expected) =>
      val diff = prediction - expected
      diff * diff
    }.sum / output.length.toDouble

  /** Derivative of mean squared error with respect to the output activations. */
  def mseDerivative(output: Vec, target: Vec): Vec =
    require(
      output.length == target.length,
      s"loss derivative dimension mismatch: ${output.length} != ${target.length}"
    )
    val scaleFactor = 2.0 / output.length.toDouble
    (output - target) * scaleFactor
