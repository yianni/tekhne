package tekhne

import tekhne.Linalg._

/** Loss functions used during training. */
object Loss:
  private val epsilon = 1e-12

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

  /** Binary cross-entropy averaged across output dimensions. */
  def binaryCrossEntropy(output: Vec, target: Vec): Double =
    require(
      output.length == target.length,
      s"loss dimension mismatch: ${output.length} != ${target.length}"
    )
    output.zip(target).map { case (prediction, expected) =>
      val clipped = clipProbability(prediction)
      -(expected * math.log(clipped) + (1.0 - expected) * math.log(1.0 - clipped))
    }.sum / output.length.toDouble

  /** Derivative of binary cross-entropy with respect to the output activations. */
  def binaryCrossEntropyDerivative(output: Vec, target: Vec): Vec =
    require(
      output.length == target.length,
      s"loss derivative dimension mismatch: ${output.length} != ${target.length}"
    )
    val scaleFactor = 1.0 / output.length.toDouble
    output.zip(target).map { case (prediction, expected) =>
      val clipped = clipProbability(prediction)
      ((clipped - expected) / (clipped * (1.0 - clipped))) * scaleFactor
    }

  def value(loss: LossFunction, output: Vec, target: Vec): Double =
    loss match
      case LossFunction.MeanSquaredError   => mse(output, target)
      case LossFunction.BinaryCrossEntropy => binaryCrossEntropy(output, target)

  def derivative(loss: LossFunction, output: Vec, target: Vec): Vec =
    loss match
      case LossFunction.MeanSquaredError   => mseDerivative(output, target)
      case LossFunction.BinaryCrossEntropy => binaryCrossEntropyDerivative(output, target)

  private def clipProbability(value: Double): Double =
    math.max(epsilon, math.min(1.0 - epsilon, value))
