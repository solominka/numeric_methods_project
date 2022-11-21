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
