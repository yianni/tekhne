package tekhne

class LinalgSuite extends munit.FunSuite:
  import Linalg.*

  test("dot product computes expected value") {
    assertEqualsDouble(dot(Vector(1.0, 2.0), Vector(3.0, 4.0)), 11.0, 1e-12)
  }

  test("matrix vector multiply computes expected value") {
    val matrix = Vector(Vector(1.0, 2.0), Vector(3.0, 4.0))
    val vector = Vector(5.0, 6.0)
    assertEquals(matVecMul(matrix, vector), Vector(17.0, 39.0))
  }

  test("outer product computes expected matrix") {
    val left  = Vector(1.0, 2.0)
    val right = Vector(3.0, 4.0)
    assertEquals(outer(left, right), Vector(Vector(3.0, 4.0), Vector(6.0, 8.0)))
  }
