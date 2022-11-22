from direct_methods import gauss_simple, gauss_choice, gauss_elimination, tridiagonal_matrix_algorithm, LU_decomposition
from iterative_methods import simple_iteration, zeidel_method

def choose_best_method(A, eps=0.01):
    size_threshold = 10 ** 2

    if len(A) < size_threshold and not _zero_diag(A):
        if _little_values_diag(A):
            if _diag_dominance(A):
                return tridiagonal_matrix_algorithm(A)

            return gauss_choice(A)

        if _diag_dominance(A):
            return LU_decomposition(A)

        return gauss_elimination(A)

    if _is_symmetric(A):
        return zeidel_method(A, eps=eps)

    return simple_iteration(A, eps=eps)


def _is_symmetric(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] != matrix[j][i]:
                return False

    return True

def _diag_dominance(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] < sum(matrix[i][:i]) + sum(matrix[i][i + 1:len(matrix)]):
            return False

    return True


def _zero_diag(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] == 0:
            return True

    return False

def _little_values_diag(matrix, eps=0.1):
    for i in range(len(matrix)):
        if matrix[i][i] < eps:
            return True

    return False