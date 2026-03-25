package tekhne

import scala.util.Random

type Vec = Vector[Double]
type Mat = Vector[Vec]

enum Activation:
  case Sigmoid
  case Tanh
  case Identity

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
        val scale      = 1.0 / math.sqrt(inputSize.toDouble)
        val weights    = Vector.fill(outputSize, inputSize)((rng.nextDouble() * 2.0 - 1.0) * scale)
        val bias       = Vector.fill(outputSize)((rng.nextDouble() * 2.0 - 1.0) * scale)
        Dense(weights, bias, activation)
      }
      .toVector

    Network(layers)

final case class TrainingConfig(
    learningRate: Double,
    epochs: Int,
    shuffleEachEpoch: Boolean = false
):
  require(learningRate > 0.0, s"learning rate must be positive, got $learningRate")
  require(epochs > 0, s"epochs must be positive, got $epochs")

final case class LayerCache(
    input: Vec,
    preActivation: Vec,
    output: Vec
)

final case class ForwardPass(
    output: Vec,
    caches: Vector[LayerCache]
)

final case class DenseGrad(
    dWeights: Mat,
    dBias: Vec
)
