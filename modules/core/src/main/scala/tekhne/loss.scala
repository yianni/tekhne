package tekhne

import tekhne.Linalg._

/** Loss functions used during training. */
object Loss:
  private val epsilon = 1e-12

  private def requireCompatible(output: Vec, target: Vec, operation: String): Unit =
    require(
      output.length == target.length,
      s"$operation dimension mismatch: ${output.length} != ${target.length}"
    )
    require(output.nonEmpty, s"$operation vectors must be non-empty")

  private[tekhne] def requireValidTargets(loss: LossFunction, target: Vec): Unit =
    loss match
      case LossFunction.MeanSquaredError   => ()
      case LossFunction.BinaryCrossEntropy =>
        require(
          target.forall(value => value.isFinite && value >= 0.0 && value <= 1.0),
          "binary cross-entropy targets must be finite and between 0.0 and 1.0"
        )

  /** Mean squared error averaged across output dimensions. */
  def mse(output: Vec, target: Vec): Double =
    requireCompatible(output, target, "loss")
    output.zip(target).map { case (prediction, expected) =>
      val diff = prediction - expected
      diff * diff
    }.sum / output.length.toDouble

  /** Derivative of mean squared error with respect to the output activations. */
  def mseDerivative(output: Vec, target: Vec): Vec =
    requireCompatible(output, target, "loss derivative")
    val scaleFactor = 2.0 / output.length.toDouble
    (output - target) * scaleFactor

  /** Binary cross-entropy averaged across output dimensions. */
  def binaryCrossEntropy(output: Vec, target: Vec): Double =
    requireCompatible(output, target, "loss")
    requireValidTargets(LossFunction.BinaryCrossEntropy, target)
    output.zip(target).map { case (prediction, expected) =>
      val clipped = clipProbability(prediction)
      -(expected * math.log(clipped) + (1.0 - expected) * math.log(1.0 - clipped))
    }.sum / output.length.toDouble

  /** Derivative of binary cross-entropy with respect to the output activations. */
  def binaryCrossEntropyDerivative(output: Vec, target: Vec): Vec =
    requireCompatible(output, target, "loss derivative")
    requireValidTargets(LossFunction.BinaryCrossEntropy, target)
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
