package tekhne

import scala.util.Random

class TrainingSuite extends munit.FunSuite:
  private val xorData = Vector(
    (Vector(0.0, 0.0), Vector(0.0)),
    (Vector(0.0, 1.0), Vector(1.0)),
    (Vector(1.0, 0.0), Vector(1.0)),
    (Vector(1.0, 1.0), Vector(0.0))
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
      new Random(42L)
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

    val trained1 = Training.train(network, xorData, config, new Random(42L))
    val trained2 = Training.train(network, xorData, config, new Random(42L))

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
      "requirement failed: shuffleEachEpoch = true requires the Training.train overload that accepts a Random"
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

    val trained1 = Training.train(network, xorData, config, new Random(42L))
    val trained2 = Training.train(network, xorData, config, new Random(7L))

    assertNotEquals(trained1, trained2)
  }
