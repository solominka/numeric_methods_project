import numpy as np

from common import solve_upper_triangular_matrix


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
