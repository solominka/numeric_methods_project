import numpy as np

from common import norm_x, norm_a, build_diagonal_dominance


def simple_iteration(extended_matrix, eps):
    assert len(extended_matrix) > 0
    assert len(extended_matrix) + 1 == len(extended_matrix[0])

    n = len(extended_matrix)
    matrix_copy = np.array(extended_matrix, dtype=float)
    assert(build_diagonal_dominance(matrix_copy))
    assert(norm_a(matrix_copy) < 1)

    x_prev = np.zeros(n)
    x_cur = np.zeros(n)
    for i in range(n):
        x_cur[i] = matrix_copy[i][n]

    while norm_x(x_prev - x_cur) >= eps:
        x_cur, x_prev = x_prev, x_cur
        for i in range(n):
            x_cur[i] = matrix_copy[i][n]
            for j in range(n):
                if j != i:
                    x_cur[i] -= matrix_copy[i][j] * x_prev[j]
            x_cur[i] /= matrix_copy[i][i]
    return x_cur


def zeidel_method(extended_matrix, eps):
    assert len(extended_matrix) > 0
    assert len(extended_matrix) + 1 == len(extended_matrix[0])

    n = len(extended_matrix)
    matrix_copy = np.array(extended_matrix, dtype=float)
    assert(build_diagonal_dominance(matrix_copy))
    assert(norm_a(matrix_copy) < 1)

    x_prev = np.zeros(n)
    x_cur = np.zeros(n)
    for i in range(n):
        x_cur[i] = matrix_copy[i][n]

    while norm_x(x_prev - x_cur) >= eps:
        x_cur, x_prev = x_prev, x_cur
        for i in range(n):
            x_cur[i] = matrix_copy[i][n]
            for j in range(n):
                if j > i:
                    x_cur[i] -= matrix_copy[i][j] * x_prev[j]
                elif j < i:
                    x_cur[i] -= matrix_copy[i][j] * x_cur[j]
            x_cur[i] /= matrix_copy[i][i]
    return x_cur
