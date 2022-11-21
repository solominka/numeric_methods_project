import unittest

from direct_methods import tridiagonal_matrix_algorithm

EPS = 0.01


class TestCases(unittest.TestCase):
    def test_tridiagonal_matrix_algorithme(self):
        matrix = [[5, 3, 0, 0, 8], [3, 6, 1, 0, 10], [0, 1, 4, -2, 3], [0, 0, 1, -3, -2]]
        ans = tridiagonal_matrix_algorithm(matrix)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 1) < EPS
        assert abs(ans[2] - 1) < EPS
        assert abs(ans[3] - 1) < EPS


if __name__ == '__main__':
    unittest.main()
