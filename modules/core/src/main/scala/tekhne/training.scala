package tekhne

import tekhne.Linalg._

import scala.util.Random

/** Training helpers built around plain stochastic gradient descent. */
object Training:
  private def trainDeterministic(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig
  ): Network =
    Iterator
      .fill(config.epochs)(())
      .foldLeft(network) { case (current, _) =>
        trainEpoch(current, data, config.learningRate)
      }

  def step(
      network: Network,
      input: Vec,
      target: Vec,
      learningRate: Double
  ): Network =
    require(learningRate > 0.0, s"learning rate must be positive, got $learningRate")

    val grads = Backprop.gradients(network, input, target)

    val updatedLayers = network.layers.zip(grads).map { case (layer, grad) =>
      val updatedWeights = layer.weights.zip(grad.dWeights).map { case (weightsRow, gradRow) =>
        weightsRow - (gradRow * learningRate)
      }
      val updatedBias    = layer.bias - (grad.dBias * learningRate)
      layer.copy(weights = updatedWeights, bias = updatedBias)
    }

    Network(updatedLayers)

  /** Applies one full pass over the dataset in its current order. */
  def trainEpoch(
      network: Network,
      data: Vector[(Vec, Vec)],
      learningRate: Double
  ): Network =
    data.foldLeft(network) { case (current, (input, target)) =>
      step(current, input, target, learningRate)
    }

  /** Trains without shuffling.
    *
    * If `shuffleEachEpoch` is enabled, use the overload that accepts a `Random`.
    */
  def train(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig
  ): Network =
    require(
      !config.shuffleEachEpoch,
      "shuffleEachEpoch = true requires the Training.train overload that accepts a Random"
    )
    require(data.nonEmpty, "training data must be non-empty")
    trainDeterministic(network, data, config)

  /** Trains for the configured number of epochs, optionally shuffling once per epoch. */
  def train(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig,
      rng: Random
  ): Network =
    require(data.nonEmpty, "training data must be non-empty")

    Iterator
      .fill(config.epochs)(())
      .foldLeft(network) { case (current, _) =>
        val epochData =
          if config.shuffleEachEpoch then rng.shuffle(data)
          else data
        trainEpoch(current, epochData, config.learningRate)
      }

  /** Computes the average dataset loss with the current network parameters. */
  def datasetLoss(network: Network, data: Vector[(Vec, Vec)]): Double =
    require(data.nonEmpty, "training data must be non-empty")
    data.map { case (input, target) => Loss.mse(Forward.predict(network, input), target) }.sum /
      data.length.toDouble
