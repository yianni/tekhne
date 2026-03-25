package tekhne

import scala.util.Random

class InitializationSuite extends munit.FunSuite:
  test("random initialization uses zero biases") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    assertEquals(network.layers.map(_.bias), Vector(Vector(0.0, 0.0, 0.0), Vector(0.0)))
  }

  test("random initialization keeps weights within glorot bounds") {
    val network = Network.random(
      layerSizes = Vector(2, 3, 1),
      activations = Vector(Activation.Tanh, Activation.Sigmoid),
      rng = new Random(42L)
    )

    val expectedLimits = Vector(
      math.sqrt(6.0 / 5.0),
      math.sqrt(6.0 / 4.0)
    )

    network.layers.zip(expectedLimits).foreach { case (layer, limit) =>
      layer.weights.flatten.foreach { weight =>
        assert(weight >= -limit)
        assert(weight <= limit)
      }
    }
  }
