package tekhne

import tekhne.Linalg._

import scala.util.Random

/** Training helpers built around plain stochastic gradient descent. */
object Training:
  private def noOpMetricsHandler(metrics: EpochMetrics): Unit = ()

  private def batchData(
      data: Vector[(Vec, Vec)],
      batchSize: Int
  ): Vector[Vector[(Vec, Vec)]] =
    data.grouped(batchSize).map(_.toVector).toVector

  private def averageGradients(grads: Vector[Vector[DenseGrad]]): Vector[DenseGrad] =
    require(grads.nonEmpty, "mini-batch gradients must be non-empty")

    val batchSize = grads.length.toDouble
    grads.transpose.map { layerGrads =>
      val averagedWeights = layerGrads.map(_.dWeights).transpose.map { rows =>
        rows.transpose.map(_.sum / batchSize)
      }
      val averagedBias    = layerGrads.map(_.dBias).transpose.map(_.sum / batchSize)
      DenseGrad(averagedWeights, averagedBias)
    }

  private def stepBatch(
      network: Network,
      batch: Vector[(Vec, Vec)],
      learningRate: Double,
      loss: LossFunction
  ): Network =
    require(batch.nonEmpty, "mini-batch must be non-empty")
    require(learningRate > 0.0, s"learning rate must be positive, got $learningRate")

    val averagedGrads = averageGradients(batch.map { case (input, target) =>
      Backprop.gradients(network, input, target, loss)
    })

    val updatedLayers = network.layers.zip(averagedGrads).map { case (layer, grad) =>
      val updatedWeights = layer.weights.zip(grad.dWeights).map { case (weightsRow, gradRow) =>
        weightsRow - (gradRow * learningRate)
      }
      val updatedBias    = layer.bias - (grad.dBias * learningRate)
      layer.copy(weights = updatedWeights, bias = updatedBias)
    }

    Network(updatedLayers)

  private def trainDeterministic(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig,
      onEpochComplete: EpochMetrics => Unit = noOpMetricsHandler
  ): Network = (1 to config.epochs)
    .foldLeft(network) { case (current, epoch) =>
      val updated = trainEpoch(current, data, config.learningRate, config.batchSize, config.loss)
      onEpochComplete(EpochMetrics(epoch, datasetLoss(updated, data, config.loss)))
      updated
    }

  /** Applies one stochastic gradient descent update for a single training example. */
  def step(
      network: Network,
      input: Vec,
      target: Vec,
      learningRate: Double,
      loss: LossFunction = LossFunction.MeanSquaredError
  ): Network =
    require(learningRate > 0.0, s"learning rate must be positive, got $learningRate")

    val grads = Backprop.gradients(network, input, target, loss)

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
      learningRate: Double,
      batchSize: Int = 1,
      loss: LossFunction = LossFunction.MeanSquaredError
  ): Network =
    require(batchSize > 0, s"batch size must be positive, got $batchSize")
    batchData(data, batchSize).foldLeft(network) { case (current, batch) =>
      stepBatch(current, batch, learningRate, loss)
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

  def train(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig,
      onEpochComplete: EpochMetrics => Unit
  ): Network =
    require(
      !config.shuffleEachEpoch,
      "shuffleEachEpoch = true requires the Training.train overload that accepts a Random"
    )
    require(data.nonEmpty, "training data must be non-empty")
    trainDeterministic(network, data, config, onEpochComplete)

  /** Trains for the configured number of epochs.
    *
    * When `shuffleEachEpoch` is enabled, `rng` controls the per-epoch dataset shuffling.
    */
  def train(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig,
      rng: Random
  ): Network =
    train(network, data, config, rng, noOpMetricsHandler)

  def train(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig,
      rng: Random,
      onEpochComplete: EpochMetrics => Unit
  ): Network =
    require(data.nonEmpty, "training data must be non-empty")

    (1 to config.epochs)
      .foldLeft(network) { case (current, epoch) =>
        val epochData =
          if config.shuffleEachEpoch then rng.shuffle(data)
          else data
        val updated   =
          trainEpoch(current, epochData, config.learningRate, config.batchSize, config.loss)
        onEpochComplete(EpochMetrics(epoch, datasetLoss(updated, data, config.loss)))
        updated
      }

  /** Computes the average dataset loss with the current network parameters. */
  def datasetLoss(
      network: Network,
      data: Vector[(Vec, Vec)],
      loss: LossFunction = LossFunction.MeanSquaredError
  ): Double =
    require(data.nonEmpty, "training data must be non-empty")
    data.map { case (input, target) =>
      Loss.value(loss, Forward.predict(network, input), target)
    }.sum /
      data.length.toDouble
