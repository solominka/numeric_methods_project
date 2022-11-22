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


def tridiagonal_matrix_algorithm(extended_matrix):
    assert len(extended_matrix) > 0
    assert len(extended_matrix) + 1 == len(extended_matrix[0])

    n = len(extended_matrix)
    pq_matrix = np.zeros((n, n+1))
    pq_matrix[0][1] = extended_matrix[0][1] / extended_matrix[0][0]
    pq_matrix[0][n] = extended_matrix[0][n] / extended_matrix[0][0]
    pq_matrix[0][0] = 1

    for i, row in enumerate(extended_matrix):
        if i + 1 < n:  # P[i]
            pq_matrix[i][i + 1] = -row[i + 1] / (-row[i] + row[i - 1] * pq_matrix[i - 1][i])

        # Q[i]
        pq_matrix[i][n] = (row[i - 1] * pq_matrix[i - 1][n] - row[n]) / (-row[i] + row[i - 1] * pq_matrix[i - 1][i])
        pq_matrix[i][i] = 1

    return solve_upper_triangular_matrix(pq_matrix)
