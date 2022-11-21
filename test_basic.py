import unittest

from direct_methods import LU_decomposition

EPS = 0.01


class TestCases(unittest.TestCase):
    def test_LU_decomposition(self):
        matrix = [[2, 1, 4, 16], [3, 2, 1, 10], [1, 3, 3, 16]]
        ans = LU_decomposition(matrix)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 2) < EPS
        assert abs(ans[2] - 3) < EPS


if __name__ == '__main__':
    unittest.main()
