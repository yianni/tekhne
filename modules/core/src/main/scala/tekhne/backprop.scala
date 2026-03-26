package tekhne

import tekhne.Linalg._

object Backprop:
  private[tekhne] def gradients(
      network: Network,
      input: Vec,
      target: Vec,
      loss: LossFunction = LossFunction.MeanSquaredError
  ): Vector[DenseGrad] =
    require(network.layers.nonEmpty, "network must contain at least one layer")

    val forwardPass = Forward.forward(network, input)
    require(
      forwardPass.output.length == target.length,
      s"target size ${target.length} does not match output size ${forwardPass.output.length}"
    )

    val outputLayerIndex = network.layers.length - 1
    val lastLayer        = network.layers(outputLayerIndex)
    val lastCache        = forwardPass.caches(outputLayerIndex)

    val outputDelta =
      Loss
        .derivative(loss, forwardPass.output, target)
        .hadamard(lastCache.preActivation.map(ActivationOps.derivativeFromZ(
          lastLayer.activation,
          _
        )))

    val initialGrads = Vector.fill(network.layers.length)(DenseGrad(Vector.empty, Vector.empty))
    val outputGrad   = DenseGrad(outer(outputDelta, lastCache.input), outputDelta)

    val filled = initialGrads.updated(outputLayerIndex, outputGrad)

    val (_, grads) = (outputLayerIndex - 1 to 0 by -1).foldLeft((outputDelta, filled)) {
      case ((nextDelta, acc), layerIndex) =>
        val layer     = network.layers(layerIndex)
        val nextLayer = network.layers(layerIndex + 1)
        val cache     = forwardPass.caches(layerIndex)

        val propagated           = matVecMul(transpose(nextLayer.weights), nextDelta)
        val activationDerivative =
          cache.preActivation.map(ActivationOps.derivativeFromZ(layer.activation, _))
        val delta                = propagated.hadamard(activationDerivative)
        val grad                 = DenseGrad(outer(delta, cache.input), delta)
        (delta, acc.updated(layerIndex, grad))
    }

    grads
