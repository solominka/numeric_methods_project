import numpy as np

from common import solve_upper_triangular_matrix, fill_by_rectangle_rule


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


def gauss_choice(extended_matrix):
    assert len(extended_matrix) > 0
    assert len(extended_matrix) + 1 == len(extended_matrix[0])

    np_matrix = np.array(extended_matrix, dtype=float)
    for i, row in enumerate(np_matrix):
        max_index = i + np.argmax(abs(np_matrix[i:, i]))
        if max_index != i:
            np_matrix[[i, max_index]] = np_matrix[[max_index, i]]
        if row[i] != 0:
            row /= float(row[i])
        fill_by_rectangle_rule(np_matrix, i)
    return solve_upper_triangular_matrix(np_matrix)
