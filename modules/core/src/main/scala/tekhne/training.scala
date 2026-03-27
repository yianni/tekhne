package tekhne

import tekhne.Linalg._

import scala.util.Random

/** Training helpers built around plain stochastic gradient descent. */
object Training:
  private def noOpMetricsHandler(metrics: EpochMetrics): Unit = ()

  private final case class TrainingRuntime(
      rng: Option[Random],
      onEpochComplete: EpochMetrics => Unit
  )

  private def requireLossCompatibility(network: Network, loss: LossFunction): Unit =
    loss match
      case LossFunction.MeanSquaredError   => ()
      case LossFunction.BinaryCrossEntropy =>
        require(
          network.layers.last.activation == Activation.Sigmoid,
          "binary cross-entropy requires a sigmoid output layer"
        )

  private def batchData(
      data: Vector[(Vec, Vec)],
      batchSize: Int
  ): Vector[Vector[(Vec, Vec)]] =
    data.grouped(batchSize).map(_.toVector).toVector

  private def datasetAccuracy(
      network: Network,
      data: Vector[(Vec, Vec)]
  ): Option[Double] =
    val binaryCompatible = data.forall { case (input, target) =>
      target.length == 1 && Forward.predict(network, input).length == 1 &&
      (target.head == 0.0 || target.head == 1.0)
    }

    if !binaryCompatible then None
    else
      val correct = data.count { case (input, target) =>
        val prediction     = Forward.predict(network, input).head
        val predictedLabel = if prediction >= 0.5 then 1.0 else 0.0
        predictedLabel == target.head
      }
      Some(correct.toDouble / data.length.toDouble)

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

  private def trainWithRuntime(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig,
      runtime: TrainingRuntime
  ): Network =
    require(data.nonEmpty, "training data must be non-empty")
    requireLossCompatibility(network, config.loss)
    require(
      !config.shuffleEachEpoch || runtime.rng.nonEmpty,
      "shuffleEachEpoch = true requires the fully explicit Training.train overload"
    )

    (1 to config.epochs)
      .foldLeft(network) { case (current, epoch) =>
        val epochData = runtime.rng match
          case Some(rng) if config.shuffleEachEpoch => rng.shuffle(data)
          case _                                    => data

        val updated =
          trainEpoch(current, epochData, config.learningRate, config.batchSize, config.loss)
        runtime.onEpochComplete(
          EpochMetrics(
            epoch = epoch,
            loss = datasetLoss(updated, data, config.loss),
            accuracy = datasetAccuracy(updated, data)
          )
        )
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
    requireLossCompatibility(network, loss)

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
    requireLossCompatibility(network, loss)
    batchData(data, batchSize).foldLeft(network) { case (current, batch) =>
      stepBatch(current, batch, learningRate, loss)
    }

  /** Trains without shuffling.
    *
    * If `shuffleEachEpoch` is enabled, use the fully explicit overload.
    */
  def train(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig
  ): Network =
    trainWithRuntime(network, data, config, TrainingRuntime(None, noOpMetricsHandler))

  /** Trains for the configured number of epochs.
    *
    * `rng` controls per-epoch dataset shuffling when enabled and `onEpochComplete` receives loss
    * snapshots after each epoch.
    */
  def train(
      network: Network,
      data: Vector[(Vec, Vec)],
      config: TrainingConfig,
      rng: Random,
      onEpochComplete: EpochMetrics => Unit
  ): Network =
    trainWithRuntime(network, data, config, TrainingRuntime(Some(rng), onEpochComplete))

  /** Computes the average dataset loss with the current network parameters. */
  def datasetLoss(
      network: Network,
      data: Vector[(Vec, Vec)],
      loss: LossFunction = LossFunction.MeanSquaredError
  ): Double =
    require(data.nonEmpty, "training data must be non-empty")
    requireLossCompatibility(network, loss)
    data.map { case (input, target) =>
      Loss.value(loss, Forward.predict(network, input), target)
    }.sum /
      data.length.toDouble
