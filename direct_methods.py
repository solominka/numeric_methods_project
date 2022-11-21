import numpy as np

from common import solve_upper_triangular_matrix


def gauss_elimination(extended_matrix):
    assert len(extended_matrix) > 0
    assert len(extended_matrix) + 1 == len(extended_matrix[0])
    np_matrix = np.array(extended_matrix, dtype=float)
    for i, cur_row in enumerate(np_matrix):
        pivot = cur_row[i]
        if pivot != 0:
            cur_row /= float(pivot)
        fill_by_rectangle_rule(np_matrix, i)

    return solve_upper_triangular_matrix(np_matrix)


def fill_by_rectangle_rule(extended_matrix, cur_iteration):
    n = len(extended_matrix)
    for i in range(cur_iteration+1, n):
        k = extended_matrix[cur_iteration][cur_iteration]
        for j in range(cur_iteration+1, n+1):
            extended_matrix[i][j] -= extended_matrix[cur_iteration][j] * extended_matrix[i][cur_iteration] / k
        extended_matrix[i][cur_iteration] = 0
