import numpy as np

from common import solve_upper_triangular_matrix, solve_lower_triangular_matrix


def LU_decomposition(extended_matrix):
    assert len(extended_matrix) > 0
    assert len(extended_matrix) + 1 == len(extended_matrix[0])

    n = len(extended_matrix)
    lu_matrix = np.array(extended_matrix, dtype=float)
    for k in range(0, n):
        for i in range(k, n):
            sum = 0
            for s in range(0, k):
                sum += lu_matrix[i, s] * lu_matrix[s, k]
            lu_matrix[i, k] -= sum
        for j in range(k + 1, n):
            sum = 0
            for s in range(0, k):
                sum += lu_matrix[k, s] * lu_matrix[s, j]
            lu_matrix[k, j] = (lu_matrix[k, j] - sum) / lu_matrix[k, k]

    y = solve_lower_triangular_matrix(lu_matrix)

    for i in range(0, n):
        lu_matrix[i, i] = 1
        lu_matrix[i, n] = y[i]
    return solve_upper_triangular_matrix(lu_matrix)
