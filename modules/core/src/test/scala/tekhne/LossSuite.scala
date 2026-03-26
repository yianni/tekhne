package tekhne

class LossSuite extends munit.FunSuite:
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
