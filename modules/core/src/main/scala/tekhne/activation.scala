package tekhne

/** Small helpers for evaluating activation functions and their derivatives. */
object ActivationOps:
  /** Applies an activation to a single scalar value. */
  def apply(activation: Activation, x: Double): Double =
    activation match
      case Activation.Sigmoid  => 1.0 / (1.0 + math.exp(-x))
      case Activation.Tanh     => math.tanh(x)
      case Activation.Identity => x

  def derivativeFromZ(activation: Activation, z: Double): Double =
    activation match
      case Activation.Sigmoid  =>
        val y = apply(activation, z)
        y * (1.0 - y)
      case Activation.Tanh     =>
        val y = math.tanh(z)
        1.0 - y * y
      case Activation.Identity =>
        1.0

  /** Applies an activation element-wise to a vector. */
  def activate(activation: Activation, values: Vec): Vec =
    values.map(apply(activation, _))
