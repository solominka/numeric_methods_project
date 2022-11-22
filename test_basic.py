import unittest

from iterative_methods import simple_iteration, zeidel_method
from direct_methods import gauss_choice, gauss_elimination, tridiagonal_matrix_algorithm, LU_decomposition

EPS = 0.001


class TestCases(unittest.TestCase):
    def test_zeidel_method(self):
        matrix = [[10, 1, 1, 12], [2, 2, 10, 14], [2, 10, 1, 13]]
        ans = zeidel_method(matrix, EPS)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 1) < EPS
        assert abs(ans[2] - 1) < EPS

    def test_simple_iteration(self):
        matrix = [[10, 1, 1, 12], [2, 2, 10, 14], [2, 10, 1, 13]]
        ans = simple_iteration(matrix, EPS)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 1) < EPS
        assert abs(ans[2] - 1) < EPS

    def test_LU_decomposition(self):
        matrix = [[2, 1, 4, 16], [3, 2, 1, 10], [1, 3, 3, 16]]
        ans = LU_decomposition(matrix)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 2) < EPS
        assert abs(ans[2] - 3) < EPS

    def test_tridiagonal_matrix_algorithm(self):
        matrix = [[5, 3, 0, 0, 8], [3, 6, 1, 0, 10], [0, 1, 4, -2, 3], [0, 0, 1, -3, -2]]
        ans = tridiagonal_matrix_algorithm(matrix)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 1) < EPS
        assert abs(ans[2] - 1) < EPS
        assert abs(ans[3] - 1) < EPS

    def test_gauss_elimination(self):
        matrix = [[2, 1, 4, 16], [3, 2, 1, 10], [1, 3, 3, 16]]
        ans = gauss_elimination(matrix)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 2) < EPS
        assert abs(ans[2] - 3) < EPS

    def test_gauss_choice(self):
        matrix = [[-3, 2.099, 6, 3.901], [10, -7, 0, 7], [5, -1, 5, 6]]
        ans = gauss_choice(matrix)
        assert abs(ans[0] - 0) < EPS
        assert abs(ans[1] + 1) < EPS
        assert abs(ans[2] - 1) < EPS


if __name__ == '__main__':
    unittest.main()
