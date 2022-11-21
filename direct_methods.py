import numpy as np

from common import solve_upper_triangular_matrix


def gauss_simple(extended_matrix):
    assert len(extended_matrix) > 0
    assert len(extended_matrix) + 1 == len(extended_matrix[0])
    np_matrix = np.array(extended_matrix, dtype=float)
    for i, cur_row in enumerate(np_matrix):
        pivot = cur_row[i]
        if pivot != 0:
            cur_row /= float(pivot)

        for next_row in np_matrix[i+1:]:
            next_row -= next_row[i] * cur_row

    return solve_upper_triangular_matrix(np_matrix)
