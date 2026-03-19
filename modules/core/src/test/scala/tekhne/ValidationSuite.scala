package tekhne

class ValidationSuite extends munit.FunSuite:
  test("dense rejects bias size mismatch") {
    interceptMessage[IllegalArgumentException](
      "requirement failed: bias size 1 must match output size 2"
    ) {
      Dense(
        weights = Vector(
          Vector(0.1, 0.2),
          Vector(0.3, 0.4)
        ),
        bias = Vector(0.0),
        activation = Activation.Tanh
      )
    }
  }

  test("network rejects incompatible adjacent layers") {
    val left = Dense(
      weights = Vector(
        Vector(0.1, 0.2),
        Vector(0.3, 0.4)
      ),
      bias = Vector(0.0, 0.0),
      activation = Activation.Tanh
    )

    val right = Dense(
      weights = Vector(
        Vector(0.5, 0.6, 0.7)
      ),
      bias = Vector(0.0),
      activation = Activation.Sigmoid
    )

    interceptMessage[IllegalArgumentException](
      "requirement failed: layer output size 2 must match next layer input size 3"
    ) {
      Network(Vector(left, right))
    }
  }

  test("training config rejects non-positive values") {
    interceptMessage[IllegalArgumentException](
      "requirement failed: learning rate must be positive, got 0.0"
    ) {
      TrainingConfig(learningRate = 0.0, epochs = 10)
    }

    interceptMessage[IllegalArgumentException](
      "requirement failed: epochs must be positive, got 0"
    ) {
      TrainingConfig(learningRate = 0.1, epochs = 0)
    }
  }
