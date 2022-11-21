import unittest

from iterative_methods import zeidel_method

EPS = 0.001


class TestCases(unittest.TestCase):
    def test_zeidel_method(self):
        matrix = [[10, 1, 1, 12], [2, 2, 10, 14], [2, 10, 1, 13]]
        ans = zeidel_method(matrix, EPS)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 1) < EPS
        assert abs(ans[2] - 1) < EPS


if __name__ == '__main__':
    unittest.main()
