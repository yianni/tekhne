package tekhne

import tekhne.Linalg._

object Forward:
  def predict(network: Network, input: Vec): Vec =
    forward(network, input).output

  def forward(network: Network, input: Vec): ForwardPass =
    val (finalOutput, caches) = network.layers.foldLeft((input, Vector.empty[LayerCache])) {
      case ((current, acc), layer) =>
        requireRectangular(layer.weights)
        if layer.weights.nonEmpty then
          require(
            layer.weights.head.length == current.length,
            s"layer input width ${layer.weights.head.length} does not match activation size ${current.length}"
          )
        require(
          layer.bias.length == layer.weights.length,
          s"bias size ${layer.bias.length} does not match layer output size ${layer.weights.length}"
        )

        val preActivation = matVecMul(layer.weights, current) + layer.bias
        val output        = ActivationOps.activate(layer.activation, preActivation)
        val cache         = LayerCache(current, preActivation, output)
        (output, acc :+ cache)
    }

    ForwardPass(finalOutput, caches)
