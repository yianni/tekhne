package tekhne

class ValidationSuite extends munit.FunSuite:
  private val sigmoidNetwork = Network(
    Vector(
      Dense(
        weights = Vector(Vector(0.1)),
        bias = Vector(0.0),
        activation = Activation.Sigmoid
      )
    )
  )

  private val singleExample = Vector((Vector(1.0), Vector(1.0)))

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

  test("training config rejects invalid values") {
    interceptMessage[IllegalArgumentException](
      "requirement failed: learning rate must be finite and positive, got 0.0"
    ) {
      TrainingConfig(learningRate = 0.0, epochs = 10)
    }

    Vector(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity).foreach { learningRate =>
      interceptMessage[IllegalArgumentException](
        s"requirement failed: learning rate must be finite and positive, got $learningRate"
      ) {
        TrainingConfig(learningRate = learningRate, epochs = 10)
      }
    }

    interceptMessage[IllegalArgumentException](
      "requirement failed: epochs must be positive, got 0"
    ) {
      TrainingConfig(learningRate = 0.1, epochs = 0)
    }

    interceptMessage[IllegalArgumentException](
      "requirement failed: batch size must be positive, got 0"
    ) {
      TrainingConfig(learningRate = 0.1, epochs = 10, batchSize = 0)
    }
  }

  test("training operations reject non-finite learning rates") {
    Vector(Double.NaN, Double.PositiveInfinity).foreach { learningRate =>
      val expectedMessage =
        s"requirement failed: learning rate must be finite and positive, got $learningRate"

      interceptMessage[IllegalArgumentException](expectedMessage) {
        Training.step(
          sigmoidNetwork,
          singleExample.head._1,
          singleExample.head._2,
          learningRate
        )
      }
      interceptMessage[IllegalArgumentException](expectedMessage) {
        Training.trainEpoch(sigmoidNetwork, singleExample, learningRate)
      }
    }
  }

  test("training operations reject empty datasets consistently") {
    val expectedMessage = "requirement failed: training data must be non-empty"

    interceptMessage[IllegalArgumentException](expectedMessage) {
      Training.train(
        sigmoidNetwork,
        Vector.empty,
        TrainingConfig(learningRate = 0.1, epochs = 1)
      )
    }
    interceptMessage[IllegalArgumentException](expectedMessage) {
      Training.trainEpoch(sigmoidNetwork, Vector.empty, learningRate = 0.1)
    }
    interceptMessage[IllegalArgumentException](expectedMessage) {
      Training.datasetLoss(sigmoidNetwork, Vector.empty)
    }
  }

  test("binary cross-entropy training rejects invalid targets") {
    interceptMessage[IllegalArgumentException](
      "requirement failed: binary cross-entropy targets must be finite and between 0.0 and 1.0"
    ) {
      Training.step(
        sigmoidNetwork,
        input = Vector(1.0),
        target = Vector(1.1),
        learningRate = 0.1,
        loss = LossFunction.BinaryCrossEntropy
      )
    }
  }

  test("binary cross-entropy requires sigmoid output for training") {
    val network = Network(
      Vector(
        Dense(
          weights = Vector(Vector(0.1, 0.2)),
          bias = Vector(0.0),
          activation = Activation.Identity
        )
      )
    )

    val config = TrainingConfig(
      learningRate = 0.1,
      epochs = 10,
      loss = LossFunction.BinaryCrossEntropy
    )

    interceptMessage[IllegalArgumentException](
      "requirement failed: binary cross-entropy requires a sigmoid output layer"
    ) {
      Training.train(network, Vector((Vector(0.0, 1.0), Vector(1.0))), config)
    }
  }

  test("binary cross-entropy requires sigmoid output for dataset loss") {
    val network = Network(
      Vector(
        Dense(
          weights = Vector(Vector(0.1, 0.2)),
          bias = Vector(0.0),
          activation = Activation.Identity
        )
      )
    )

    interceptMessage[IllegalArgumentException](
      "requirement failed: binary cross-entropy requires a sigmoid output layer"
    ) {
      Training.datasetLoss(
        network,
        Vector((Vector(0.0, 1.0), Vector(1.0))),
        LossFunction.BinaryCrossEntropy
      )
    }
  }
