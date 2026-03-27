package tekhne.demo

import tekhne._

import scala.util.Random

@main def runTekhne(): Unit =
  val data = Vector(
    (Vector(0.0, 0.0), Vector(0.0)),
    (Vector(0.0, 1.0), Vector(1.0)),
    (Vector(1.0, 0.0), Vector(1.0)),
    (Vector(1.0, 1.0), Vector(0.0))
  )

  val network = Network.random(
    layerSizes = Vector(2, 3, 1),
    activations = Vector(Activation.Tanh, Activation.Sigmoid),
    rng = new Random(42L)
  )

  val initialLoss = Training.datasetLoss(network, data)
  println(f"initial loss: $initialLoss%.6f")

  val trained = Training.train(
    network,
    data,
    TrainingConfig(
      learningRate = 0.1,
      epochs = 50_000
    ),
    new Random(42L),
    metrics =>
      if metrics.epoch % 10_000 == 0 then
        val accuracy = metrics.accuracy.fold("n/a")(value => f"$value%.3f")
        println(f"epoch ${metrics.epoch}%5d loss=${metrics.loss}%.6f accuracy=$accuracy")
  )

  val finalLoss = Training.datasetLoss(trained, data)
  println(f"final loss:   $finalLoss%.6f")

  data.foreach { case (input, target) =>
    val prediction = Forward.predict(trained, input)
    println(s"input=$input target=$target prediction=$prediction")
  }
