package tekhne

class ForwardSuite extends munit.FunSuite:
  test("forward pass returns expected output for known network") {
    val network = Network(
      Vector(
        Dense(
          weights = Vector(Vector(1.0, -1.0)),
          bias = Vector(0.5),
          activation = Activation.Identity
        )
      )
    )

    val result = Forward.forward(network, Vector(2.0, 1.0))

    assertEquals(result.output, Vector(1.5))
    assertEquals(result.caches.length, 1)
    assertEquals(result.caches.head.input, Vector(2.0, 1.0))
    assertEquals(result.caches.head.preActivation, Vector(1.5))
    assertEquals(result.caches.head.output, Vector(1.5))
  }
