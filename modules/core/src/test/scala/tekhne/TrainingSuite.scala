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
