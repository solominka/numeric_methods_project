import unittest

from direct_methods import gauss_choice, gauss_elimination

EPS = 0.01


class TestCases(unittest.TestCase):
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
