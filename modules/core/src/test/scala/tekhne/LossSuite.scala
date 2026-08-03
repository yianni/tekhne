package tekhne

class LossSuite extends munit.FunSuite:
  private def assertRejectsEmpty(operation: String)(evaluate: => Unit): Unit =
    interceptMessage[IllegalArgumentException](
      s"requirement failed: $operation vectors must be non-empty"
    )(evaluate)

  test("binary cross-entropy returns expected value for simple case") {
    val output = Vector(0.9)
    val target = Vector(1.0)

    val loss = Loss.binaryCrossEntropy(output, target)

    assertEqualsDouble(loss, -math.log(0.9), 1e-12)
  }

  test("binary cross-entropy derivative matches expected value") {
    val output = Vector(0.9)
    val target = Vector(1.0)

    val derivative = Loss.binaryCrossEntropyDerivative(output, target)

    assertEqualsDouble(derivative.head, -1.0 / 0.9, 1e-12)
  }

  test("binary cross-entropy rejects mismatched dimensions") {
    interceptMessage[IllegalArgumentException](
      "requirement failed: loss dimension mismatch: 1 != 2"
    ) {
      Loss.binaryCrossEntropy(Vector(0.9), Vector(1.0, 0.0))
    }
  }

  test("loss functions reject empty vectors") {
    assertRejectsEmpty("loss") {
      Loss.mse(Vector.empty, Vector.empty)
    }
    assertRejectsEmpty("loss") {
      Loss.binaryCrossEntropy(Vector.empty, Vector.empty)
    }
    assertRejectsEmpty("loss derivative") {
      Loss.mseDerivative(Vector.empty, Vector.empty)
    }
    assertRejectsEmpty("loss derivative") {
      Loss.binaryCrossEntropyDerivative(Vector.empty, Vector.empty)
    }
  }

  test("binary cross-entropy rejects invalid targets") {
    Vector(Double.NaN, Double.PositiveInfinity, -0.1, 1.1).foreach { invalidTarget =>
      val expectedMessage =
        "requirement failed: binary cross-entropy targets must be finite and between 0.0 and 1.0"

      interceptMessage[IllegalArgumentException](expectedMessage) {
        Loss.binaryCrossEntropy(Vector(0.5), Vector(invalidTarget))
      }
      interceptMessage[IllegalArgumentException](expectedMessage) {
        Loss.binaryCrossEntropyDerivative(Vector(0.5), Vector(invalidTarget))
      }
    }
  }
