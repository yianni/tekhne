package tekhne

class ActivationSuite extends munit.FunSuite:
  test("sigmoid output stays between zero and one") {
    val result = ActivationOps(Activation.Sigmoid, 0.0)
    assert(result > 0.0)
    assert(result < 1.0)
    assertEqualsDouble(result, 0.5, 1e-12)
  }

  test("identity derivative is one") {
    assertEqualsDouble(ActivationOps.derivativeFromZ(Activation.Identity, 123.0), 1.0, 1e-12)
  }

  test("tanh derivative at zero is one") {
    assertEqualsDouble(ActivationOps.derivativeFromZ(Activation.Tanh, 0.0), 1.0, 1e-12)
  }
