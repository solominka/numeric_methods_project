import unittest

from direct_methods import gauss_simple

EPS = 0.01


class TestCases(unittest.TestCase):
    def test_gauss_simple(self):
        matrix = [[5, 0, 1, 11], [2, 6, -2, 5], [-3, 2, 10, 6]]
        ans = gauss_simple(matrix)
        assert abs(ans[0] - 1.98) < EPS
        assert abs(ans[1] - 0.54) < EPS
        assert abs(ans[2] - 1.09) < EPS


if __name__ == '__main__':
    unittest.main()
