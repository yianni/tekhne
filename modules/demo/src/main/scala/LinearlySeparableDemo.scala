package tekhne.demo

import tekhne._

import scala.util.Random

@main def runLinearlySeparableDemo(): Unit =
  val data = Vector(
    (Vector(0.0, 0.1), Vector(0.0)),
    (Vector(0.1, 0.2), Vector(0.0)),
    (Vector(0.2, 0.1), Vector(0.0)),
    (Vector(0.3, 0.2), Vector(0.0)),
    (Vector(0.7, 0.8), Vector(1.0)),
    (Vector(0.8, 0.9), Vector(1.0)),
    (Vector(0.9, 0.8), Vector(1.0)),
    (Vector(0.8, 0.7), Vector(1.0))
  )

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

  val initialLoss = Training.datasetLoss(network, data, config.loss)
  println("Linearly separable demo")
  println(f"initial loss: $initialLoss%.6f")

  val trained = Training.train(
    network,
    data,
    config,
    new Random(42L),
    metrics =>
      if metrics.epoch % 500 == 0 then
        println(f"epoch ${metrics.epoch}%4d loss=${metrics.loss}%.6f")
  )

  val finalLoss = Training.datasetLoss(trained, data, config.loss)
  println(f"final loss:   $finalLoss%.6f")

  data.foreach { case (input, target) =>
    val prediction = Forward.predict(trained, input)
    println(s"input=$input target=$target prediction=$prediction")
  }
