package tekhne

import scala.annotation.targetName

object Linalg:
  def dot(left: Vec, right: Vec): Double =
    require(
      left.length == right.length,
      s"dot dimension mismatch: ${left.length} != ${right.length}"
    )
    left.zip(right).map(_ * _).sum

  def add(left: Vec, right: Vec): Vec =
    require(
      left.length == right.length,
      s"add dimension mismatch: ${left.length} != ${right.length}"
    )
    left.zip(right).map(_ + _)

  def sub(left: Vec, right: Vec): Vec =
    require(
      left.length == right.length,
      s"sub dimension mismatch: ${left.length} != ${right.length}"
    )
    left.zip(right).map(_ - _)

  def scale(vector: Vec, scalar: Double): Vec =
    vector.map(_ * scalar)

  def hadamard(left: Vec, right: Vec): Vec =
    require(
      left.length == right.length,
      s"hadamard dimension mismatch: ${left.length} != ${right.length}"
    )
    left.zip(right).map(_ * _)

  def matVecMul(matrix: Mat, vector: Vec): Vec =
    requireRectangular(matrix)
    if matrix.nonEmpty then
      require(
        matrix.head.length == vector.length,
        s"matVec dimension mismatch: ${matrix.head.length} != ${vector.length}"
      )
    matrix.map(row => dot(row, vector))

  def transpose(matrix: Mat): Mat =
    requireRectangular(matrix)
    if matrix.isEmpty then Vector.empty
    else matrix.head.indices.map(column => matrix.map(_(column))).toVector

  def outer(left: Vec, right: Vec): Mat =
    left.map(leftValue => right.map(rightValue => leftValue * rightValue))

  def requireRectangular(matrix: Mat): Unit =
    val rowWidths = matrix.map(_.length).distinct
    require(rowWidths.length <= 1, s"matrix rows must have equal width, found $rowWidths")

  extension (left: Vec)
    def +(right: Vec): Vec        = add(left, right)
    def -(right: Vec): Vec        = sub(left, right)
    def *(scalar: Double): Vec    = scale(left, scalar)
    @targetName("hadamardExt")
    def hadamard(right: Vec): Vec = Linalg.hadamard(left, right)
