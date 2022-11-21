import unittest

from iterative_methods import simple_iteration

EPS = 0.001


class TestCases(unittest.TestCase):
    def test_simple_iteration(self):
        matrix = [[10, 1, 1, 12], [2, 2, 10, 14], [2, 10, 1, 13]]
        ans = simple_iteration(matrix, EPS)
        assert abs(ans[0] - 1) < EPS
        assert abs(ans[1] - 1) < EPS
        assert abs(ans[2] - 1) < EPS


if __name__ == '__main__':
    unittest.main()
