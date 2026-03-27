package tekhne

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class TrainingSuite extends munit.FunSuite:
  private val xorData = Vector(
    (Vector(0.0, 0.0), Vector(0.0)),
    (Vector(0.0, 1.0), Vector(1.0)),
    (Vector(1.0, 0.0), Vector(1.0)),
    (Vector(1.0, 1.0), Vector(0.0))
  )

  private val linearlySeparableData = Vector(
    (Vector(0.0, 0.1), Vector(0.0)),
    (Vector(0.1, 0.2), Vector(0.0)),
    (Vector(0.2, 0.1), Vector(0.0)),
    (Vector(0.3, 0.2), Vector(0.0)),
    (Vector(0.7, 0.8), Vector(1.0)),
    (Vector(0.8, 0.9), Vector(1.0)),
    (Vector(0.9, 0.8), Vector(1.0)),
    (Vector(0.8, 0.7), Vector(1.0))
  )

  test("training lowers xor loss") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val initialLoss = Training.datasetLoss(network, xorData)

    val trained = Training.train(
      network,
      xorData,
      TrainingConfig(
        learningRate = 0.1,
        epochs = 50_000
      )
    )

    val finalLoss = Training.datasetLoss(trained, xorData)

    assert(finalLoss < initialLoss)
    assert(finalLoss < 0.02)
  }

  test("trained xor network separates classes") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val trained = Training.train(
      network,
      xorData,
      TrainingConfig(
        learningRate = 0.1,
        epochs = 50_000
      )
    )

    val predictions = xorData.map { case (input, _) =>
      Forward.predict(trained, input).head
    }

    assert(predictions(0) < 0.1)
    assert(predictions(1) > 0.9)
    assert(predictions(2) > 0.9)
    assert(predictions(3) < 0.1)
  }

  test("shuffled training lowers xor loss") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val initialLoss = Training.datasetLoss(network, xorData)

    val trained = Training.train(
      network,
      xorData,
      TrainingConfig(
        learningRate = 0.1,
        epochs = 50_000,
        shuffleEachEpoch = true
      ),
      new Random(42L),
      _ => ()
    )

    val finalLoss = Training.datasetLoss(trained, xorData)

    assert(finalLoss < initialLoss)
    assert(finalLoss < 0.02)
  }

  test("shuffled training is deterministic with a fixed seed") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 50_000,
      shuffleEachEpoch = true
    )

    val trained1 = Training.train(network, xorData, config, new Random(42L), _ => ())
    val trained2 = Training.train(network, xorData, config, new Random(42L), _ => ())

    assertEquals(trained1, trained2)
  }

  test("default training rejects shuffling without explicit rng") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 50_000,
      shuffleEachEpoch = true
    )

    interceptMessage[IllegalArgumentException](
      "requirement failed: shuffleEachEpoch = true requires the fully explicit Training.train overload"
    ) {
      Training.train(network, xorData, config)
    }
  }

  test("shuffled training can vary across seeds") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 50_000,
      shuffleEachEpoch = true
    )

    val trained1 = Training.train(network, xorData, config, new Random(42L), _ => ())
    val trained2 = Training.train(network, xorData, config, new Random(7L), _ => ())

    assertNotEquals(trained1, trained2)
  }

  test("batch size one matches existing training behavior") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 50_000,
      batchSize = 1
    )

    val trainedWithDefault  = Training.train(network, xorData, TrainingConfig(0.1, 50_000))
    val trainedWithBatching = Training.train(network, xorData, config)

    assertEquals(trainedWithBatching, trainedWithDefault)
  }

  test("mini-batch training lowers xor loss") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val initialLoss = Training.datasetLoss(network, xorData)

    val trained = Training.train(
      network,
      xorData,
      TrainingConfig(
        learningRate = 0.1,
        epochs = 50_000,
        batchSize = 2
      )
    )

    val finalLoss = Training.datasetLoss(trained, xorData)

    assert(finalLoss < initialLoss)
    assert(finalLoss < 0.05)
  }

  test("mini-batch training still separates xor classes") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val trained = Training.train(
      network,
      xorData,
      TrainingConfig(
        learningRate = 0.1,
        epochs = 50_000,
        batchSize = 2
      )
    )

    val predictions = xorData.map { case (input, _) =>
      Forward.predict(trained, input).head
    }

    assert(predictions(0) < 0.2)
    assert(predictions(1) > 0.8)
    assert(predictions(2) > 0.8)
    assert(predictions(3) < 0.2)
  }

  test("shuffled mini-batch training is deterministic with a fixed seed") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 50_000,
      shuffleEachEpoch = true,
      batchSize = 2
    )

    val trained1 = Training.train(network, xorData, config, new Random(42L), _ => ())
    val trained2 = Training.train(network, xorData, config, new Random(42L), _ => ())

    assertEquals(trained1, trained2)
  }

  test("binary cross-entropy training lowers xor loss") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 50_000,
      batchSize = 2,
      loss = LossFunction.BinaryCrossEntropy
    )

    val initialLoss = Training.datasetLoss(network, xorData, config.loss)
    val trained     = Training.train(network, xorData, config)
    val finalLoss   = Training.datasetLoss(trained, xorData, config.loss)

    assert(finalLoss < initialLoss)
    assert(finalLoss < 0.02)
  }

  test("binary cross-entropy training still separates xor classes") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val trained = Training.train(
      network,
      xorData,
      TrainingConfig(
        learningRate = 0.1,
        epochs = 50_000,
        batchSize = 2,
        loss = LossFunction.BinaryCrossEntropy
      )
    )

    val predictions = xorData.map { case (input, _) =>
      Forward.predict(trained, input).head
    }

    assert(predictions(0) < 0.1)
    assert(predictions(1) > 0.9)
    assert(predictions(2) > 0.9)
    assert(predictions(3) < 0.1)
  }

  test("training callback is invoked once per epoch with increasing epoch numbers") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 4,
      batchSize = 2
    )

    val observed = ArrayBuffer.empty[EpochMetrics]

    Training.train(network, xorData, config, new Random(0L), metrics => observed += metrics)

    assertEquals(observed.map(_.epoch).toVector, Vector(1, 2, 3, 4))
    assert(observed.forall(metrics => metrics.loss.isFinite))
    assert(observed.forall(_.accuracy.isDefined))
  }

  test("reported losses decrease during BCE training") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 10,
      batchSize = 2,
      loss = LossFunction.BinaryCrossEntropy
    )

    val observed = ArrayBuffer.empty[EpochMetrics]

    Training.train(network, xorData, config, new Random(0L), metrics => observed += metrics)

    assertEquals(observed.length, 10)
    assert(observed.last.loss < observed.head.loss)
    assert(observed.forall(_.accuracy.isDefined))
    assert(observed.last.accuracy.get >= observed.head.accuracy.get)
    assert(observed.last.accuracy.get >= 0.75)
  }

  test("metrics callback works with shuffled training when rng is provided") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 5,
      shuffleEachEpoch = true,
      batchSize = 2
    )

    val observed = ArrayBuffer.empty[EpochMetrics]

    Training.train(network, xorData, config, new Random(42L), metrics => observed += metrics)

    assertEquals(observed.map(_.epoch).toVector, Vector(1, 2, 3, 4, 5))
    assert(observed.forall(metrics => metrics.loss.isFinite))
    assert(observed.forall(_.accuracy.isDefined))
  }

  test("accuracy is absent when targets are not binary-compatible") {
    val regressionData = Vector(
      (Vector(0.0), Vector(0.25)),
      (Vector(1.0), Vector(0.75))
    )

    val network = Network.random(
      layerSizes = Vector(1, 1),
      activations = Vector(Activation.Sigmoid),
      rng = new Random(42L)
    )

    val observed = ArrayBuffer.empty[EpochMetrics]

    Training.train(
      network,
      regressionData,
      TrainingConfig(learningRate = 0.1, epochs = 2),
      new Random(0L),
      metrics => observed += metrics
    )

    assertEquals(observed.length, 2)
    assert(observed.forall(_.accuracy.isEmpty))
  }

  test("linearly separable dataset learns successfully") {
    val network = Network.random(
      layerSizes = Vector(2, 1),
      activations = Vector(Activation.Sigmoid),
      rng = new Random(42L)
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 5_000,
      shuffleEachEpoch = true,
      batchSize = 2,
      loss = LossFunction.BinaryCrossEntropy
    )

    val initialLoss = Training.datasetLoss(network, linearlySeparableData, config.loss)
    val trained     = Training.train(network, linearlySeparableData, config, new Random(42L), _ => ())
    val finalLoss   = Training.datasetLoss(trained, linearlySeparableData, config.loss)
    val predictions = linearlySeparableData.map { case (input, _) =>
      Forward.predict(trained, input).head
    }

    assert(finalLoss < initialLoss)
    assert(finalLoss < 0.02)
    assert(predictions.take(4).forall(_ < 0.1))
    assert(predictions.drop(4).forall(_ > 0.9))
  }
