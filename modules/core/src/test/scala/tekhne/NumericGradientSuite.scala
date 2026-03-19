package tekhne

class NumericGradientSuite extends munit.FunSuite:
  private val epsilon = 1e-6

  test("backprop gradients match numeric gradients for a small network") {
    val network = Network(
      Vector(
        Dense(
          weights = Vector(
            Vector(0.15, -0.2),
            Vector(0.4, 0.1)
          ),
          bias = Vector(0.05, -0.03),
          activation = Activation.Tanh
        ),
        Dense(
          weights = Vector(
            Vector(0.25, -0.35)
          ),
          bias = Vector(0.12),
          activation = Activation.Sigmoid
        )
      )
    )

    val input  = Vector(0.7, -0.4)
    val target = Vector(0.8)

    val analytic = Backprop.gradients(network, input, target)
    val numeric  = numericGradients(network, input, target)

    compareGradients(analytic, numeric, tolerance = 1e-4)
  }

  private def numericGradients(network: Network, input: Vec, target: Vec): Vector[DenseGrad] =
    network.layers.zipWithIndex.map { case (layer, layerIndex) =>
      val dWeights = layer.weights.zipWithIndex.map { case (row, rowIndex) =>
        row.indices.map { colIndex =>
          centralDifference(network, input, target)((current, delta) =>
            updateWeight(current, layerIndex, rowIndex, colIndex)(delta)
          )
        }.toVector
      }

      val dBias = layer.bias.indices.map { biasIndex =>
        centralDifference(network, input, target)((current, delta) =>
          updateBias(current, layerIndex, biasIndex)(delta)
        )
      }.toVector

      DenseGrad(dWeights, dBias)
    }

  private def centralDifference(
      network: Network,
      input: Vec,
      target: Vec
  )(
      update: (Network, Double) => Network
  ): Double =
    val plus  = Loss.mse(Forward.predict(update(network, epsilon), input), target)
    val minus = Loss.mse(Forward.predict(update(network, -epsilon), input), target)
    (plus - minus) / (2.0 * epsilon)

  private def updateWeight(
      network: Network,
      layerIndex: Int,
      rowIndex: Int,
      colIndex: Int
  )(
      delta: Double
  ): Network =
    val updatedLayers = network.layers.updated(
      layerIndex,
      network.layers(layerIndex).copy(
        weights = network.layers(layerIndex).weights.updated(
          rowIndex,
          network.layers(layerIndex).weights(rowIndex).updated(
            colIndex,
            network.layers(layerIndex).weights(rowIndex)(colIndex) + delta
          )
        )
      )
    )
    Network(updatedLayers)

  private def updateBias(
      network: Network,
      layerIndex: Int,
      biasIndex: Int
  )(delta: Double): Network =
    val updatedLayers = network.layers.updated(
      layerIndex,
      network.layers(layerIndex).copy(
        bias = network.layers(layerIndex).bias.updated(
          biasIndex,
          network.layers(layerIndex).bias(biasIndex) + delta
        )
      )
    )
    Network(updatedLayers)

  private def compareGradients(
      analytic: Vector[DenseGrad],
      numeric: Vector[DenseGrad],
      tolerance: Double
  ): Unit =
    assertEquals(analytic.length, numeric.length)

    analytic.zip(numeric).zipWithIndex.foreach { case ((analyticLayer, numericLayer), layerIndex) =>
      analyticLayer.dWeights.zip(numericLayer.dWeights).zipWithIndex.foreach {
        case ((analyticRow, numericRow), rowIndex) =>
          analyticRow.zip(numericRow).zipWithIndex.foreach {
            case ((analyticValue, numericValue), colIndex) =>
              assert(
                math.abs(analyticValue - numericValue) <= tolerance,
                clues(
                  s"weight gradient mismatch at layer=$layerIndex row=$rowIndex col=$colIndex",
                  s"analytic=$analyticValue numeric=$numericValue tolerance=$tolerance"
                )
              )
          }
      }

      analyticLayer.dBias.zip(numericLayer.dBias).zipWithIndex.foreach {
        case ((analyticValue, numericValue), biasIndex) =>
          assert(
            math.abs(analyticValue - numericValue) <= tolerance,
            clues(
              s"bias gradient mismatch at layer=$layerIndex index=$biasIndex",
              s"analytic=$analyticValue numeric=$numericValue tolerance=$tolerance"
            )
          )
      }
    }
