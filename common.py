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


def solve_upper_triangular_matrix(extended_matrix):
    n = len(extended_matrix)
    solution = [0] * n
    for i in range(n-1, -1, -1):
        solution[i] = extended_matrix[i][n]
        for j in range(i+1, n):
            solution[i] -= extended_matrix[i][j] * solution[j]
    return solution


def solve_lower_triangular_matrix(extended_matrix):
    n = len(extended_matrix)
    solution = [0] * n
    for i in range(0, n):
        solution[i] = extended_matrix[i][n]
        for j in range(0, i):
            solution[i] -= extended_matrix[i][j] * solution[j]
        solution[i] /= extended_matrix[i][i]
    return solution


def fill_by_rectangle_rule(extended_matrix, cur_iteration):
    n = len(extended_matrix)
    for i in range(cur_iteration+1, n):
        k = extended_matrix[cur_iteration][cur_iteration]
        for j in range(cur_iteration+1, n+1):
            extended_matrix[i][j] -= extended_matrix[cur_iteration][j] * extended_matrix[i][cur_iteration] / k
        extended_matrix[i][cur_iteration] = 0
