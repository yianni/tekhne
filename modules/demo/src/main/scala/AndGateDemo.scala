package tekhne.demo

import tekhne._

import scala.util.Random

@main def runAndGateDemo(): Unit =
  val data = Vector(
    (Vector(0.0, 0.0), Vector(0.0)),
    (Vector(0.0, 1.0), Vector(0.0)),
    (Vector(1.0, 0.0), Vector(0.0)),
    (Vector(1.0, 1.0), Vector(1.0))
  )

  val network = Network.random(
    layerSizes = Vector(2, 1),
    activations = Vector(Activation.Sigmoid),
    rng = new Random(7L)
  )

  val trained = Training.train(
    network,
    data,
    TrainingConfig(
      learningRate = 0.2,
      epochs = 20_000
    ),
    new Random(7L),
    metrics =>
      if metrics.epoch % 5_000 == 0 then
        val accuracy = metrics.accuracy.fold("n/a")(value => f"$value%.3f")
        println(f"epoch ${metrics.epoch}%5d loss=${metrics.loss}%.6f accuracy=$accuracy")
  )

  println("AND gate demo")
  data.foreach { case (input, target) =>
    val prediction = Forward.predict(trained, input)
    println(s"input=$input target=$target prediction=$prediction")
  }
