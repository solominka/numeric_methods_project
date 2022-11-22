def solve_upper_triangular_matrix(extended_matrix):
    n = len(extended_matrix)
    solution = [0] * n
    for i in range(n-1, -1, -1):
        solution[i] = extended_matrix[i][n]
        for j in range(i+1, n):
            solution[i] -= extended_matrix[i][j] * solution[j]
    return solution

def fill_by_rectangle_rule(extended_matrix, cur_iteration):
    n = len(extended_matrix)
    for i in range(cur_iteration+1, n):
        k = extended_matrix[cur_iteration][cur_iteration]
        for j in range(cur_iteration+1, n+1):
            extended_matrix[i][j] -= extended_matrix[cur_iteration][j] * extended_matrix[i][cur_iteration] / k
        extended_matrix[i][cur_iteration] = 0
