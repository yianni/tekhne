package tekhne

import scala.util.Random

type Vec = Vector[Double]
type Mat = Vector[Vec]

/** Supported activation functions for dense layers. */
enum Activation:
  case Sigmoid
  case Tanh
  case Identity

/** Supported loss functions for training and evaluation. */
enum LossFunction:
  case MeanSquaredError
  case BinaryCrossEntropy

/** A fully connected layer with a weight matrix, bias vector, and activation.
  *
  * The matrix shape is `outputSize x inputSize`.
  */
final case class Dense(
    weights: Mat,
    bias: Vec,
    activation: Activation
):
  Linalg.requireRectangular(weights)
  require(weights.nonEmpty, "dense layer must have at least one output neuron")
  require(weights.head.nonEmpty, "dense layer must have at least one input feature")
  require(
    bias.length == weights.length,
    s"bias size ${bias.length} must match output size ${weights.length}"
  )

/** An immutable feed-forward network made of dense layers. */
final case class Network(layers: Vector[Dense]):
  require(layers.nonEmpty, "network must contain at least one layer")
  layers.sliding(2).foreach {
    case Vector(left, right) =>
      require(
        left.weights.length == right.weights.head.length,
        s"layer output size ${left.weights.length} must match next layer input size ${right.weights.head.length}"
      )
    case _                   => ()
  }

object Network:
  private def glorotLimit(inputSize: Int, outputSize: Int): Double =
    math.sqrt(6.0 / (inputSize + outputSize).toDouble)

  /** Builds a randomly initialized dense network from adjacent layer sizes.
    *
    * Weights use a small Glorot-style uniform initialization and biases start at zero.
    */
  def random(
      layerSizes: Vector[Int],
      activations: Vector[Activation],
      rng: Random
  ): Network =
    require(layerSizes.length >= 2, "network must have at least an input and output layer")
    require(
      activations.length == layerSizes.length - 1,
      "there must be exactly one activation per dense layer"
    )
    require(layerSizes.forall(_ > 0), "all layer sizes must be positive")

    val layers = layerSizes
      .sliding(2)
      .zip(activations)
      .map { case (sizes, activation) =>
        val inputSize  = sizes.head
        val outputSize = sizes(1)
        val limit      = glorotLimit(inputSize, outputSize)
        val weights    = Vector.fill(outputSize, inputSize)((rng.nextDouble() * 2.0 - 1.0) * limit)
        val bias       = Vector.fill(outputSize)(0.0)
        Dense(weights, bias, activation)
      }
      .toVector

    Network(layers)

/** Training settings for gradient descent.
  *
  * When `shuffleEachEpoch` is enabled, use the `Training.train` overload that accepts a `Random`.
  */
final case class TrainingConfig(
    learningRate: Double,
    epochs: Int,
    shuffleEachEpoch: Boolean = false,
    batchSize: Int = 1,
    loss: LossFunction = LossFunction.MeanSquaredError
):
  require(learningRate > 0.0, s"learning rate must be positive, got $learningRate")
  require(epochs > 0, s"epochs must be positive, got $epochs")
  require(batchSize > 0, s"batch size must be positive, got $batchSize")

/** Loss snapshot reported after an epoch completes. */
final case class EpochMetrics(
    epoch: Int,
    loss: Double
)

private[tekhne] final case class LayerCache(
    input: Vec,
    preActivation: Vec,
    output: Vec
)

private[tekhne] final case class ForwardPass(
    output: Vec,
    caches: Vector[LayerCache]
)

private[tekhne] final case class DenseGrad(
    dWeights: Mat,
    dBias: Vec
)
