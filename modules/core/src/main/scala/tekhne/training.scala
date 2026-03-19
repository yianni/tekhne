package tekhne

import tekhne.Linalg._

object Training:
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

  def trainEpoch(
      network: Network,
      data: Vector[(Vec, Vec)],
      learningRate: Double
  ): Network =
    data.foldLeft(network) { case (current, (input, target)) =>
      step(current, input, target, learningRate)
    }

  def train(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig
  ): Network =
    require(
      config.learningRate > 0.0,
      s"learning rate must be positive, got ${config.learningRate}"
    )
    require(config.epochs > 0, s"epochs must be positive, got ${config.epochs}")
    require(data.nonEmpty, "training data must be non-empty")

    Iterator
      .fill(config.epochs)(())
      .foldLeft(network) { case (current, _) =>
        trainEpoch(current, data, config.learningRate)
      }

  def datasetLoss(network: Network, data: Vector[(Vec, Vec)]): Double =
    require(data.nonEmpty, "training data must be non-empty")
    data.map { case (input, target) => Loss.mse(Forward.predict(network, input), target) }.sum /
      data.length.toDouble
