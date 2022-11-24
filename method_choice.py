from common import build_diagonal_dominance
from direct_methods import gauss_simple, gauss_choice, gauss_elimination, \
        tridiagonal_matrix_algorithm, LU_decomposition
from iterative_methods import simple_iteration, zeidel_method
import numpy as np

def choose_best_method(extended_matrix, method_eps=0.01):
    assert len(extended_matrix) > 0
    assert len(extended_matrix) + 1 == len(extended_matrix[0])
    
    size_threshold = 100

    extended_matrix = np.array(extended_matrix, dtype=float)

    if len(extended_matrix) < size_threshold and not _zero_diag(extended_matrix):
        
        if _little_values_diag(extended_matrix):
            if build_diagonal_dominance(extended_matrix):
                return tridiagonal_matrix_algorithm(extended_matrix)

            return gauss_choice(extended_matrix)

        if build_diagonal_dominance(extended_matrix):
            return LU_decomposition(extended_matrix)

        return gauss_elimination(extended_matrix)

    if _is_symmetric(extended_matrix):
        return zeidel_method(extended_matrix, eps=method_eps)

    return simple_iteration(extended_matrix, eps=method_eps)


def _is_symmetric(matrix, eps=10e-9):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if abs(matrix[i][j] - matrix[j][i]) < eps:
                return False

    return True


def _zero_diag(matrix, eps=10e-9):
    for i in range(len(matrix)):
        if abs(matrix[i][i]) < eps:
            return True

    return False


def _little_values_diag(matrix, eps=0.1):
    for i in range(len(matrix)):
        if abs(matrix[i][i]) < eps:
            return True

    return False