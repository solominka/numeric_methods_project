import numpy as np


def norm_x(vector):
    return np.max(abs(vector))


def norm_a(extended_matrix):
    max_sum = 0
    for row in extended_matrix:
        max_sum = max(max_sum, sum(abs(row)))

    extended_matrix /= (max_sum+1)
    return max_sum / (max_sum + 1)


def build_diagonal_dominance(extended_matrix):
    n = len(extended_matrix)
    row_dominant = [0] * n

    for i, row in enumerate(extended_matrix):
        row_sum = sum(abs(row[:-1]))
        row_max_ind = int(np.argmax(abs(row[:-1])))
        if row_sum - abs(row[row_max_ind]) >= abs(row[row_max_ind]):
            return False
        row_dominant[i] = row_max_ind

    for i, row in enumerate(extended_matrix):
        if row_dominant[i] < i:
            if row_dominant[row_dominant[i]] == i:
                swap_lines(extended_matrix, row_dominant[i], i)
            else:
                return False
    return True


def swap_lines(extended_matrix, i, j):
    tmp = extended_matrix[i].copy()
    extended_matrix[i] = extended_matrix[j].copy()
    extended_matrix[j] = tmp
