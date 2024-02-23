import time
"""
This module finds the solution of a given sudoku problem
Code credits: Tim Ruscica
More info: https://techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/
Example input board
board = [
    [7,8,0,4,0,0,1,2,0],
    [6,0,0,0,7,5,0,0,9],
    [0,0,0,6,0,1,0,7,8],
    [0,0,7,0,4,0,2,6,0],
    [0,0,1,0,5,0,9,3,0],
    [9,0,4,0,6,0,0,0,5],
    [0,7,0,3,0,0,0,1,2],
    [1,2,0,0,0,7,4,0,0],
    [0,4,9,2,0,6,0,0,7]
]
"""

def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            if solve(bo):
                return True
            bo[row][col] = 0
    return False

def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False
    return True

def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")

def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col
    return None



# numbers1= [0, 1, 9, 0, 0, 6, 0, 0, 0, 2, 0, 8, 3, 1, 0, 5, 0, 6, 0, 6, 0, 0,
#        7, 0, 0, 1, 0, 9, 3, 5, 6, 0, 1, 7, 2, 4, 1, 8, 7, 0, 0, 4, 0, 5,
#        3, 0, 2, 4, 8, 0, 0, 0, 0, 9, 8, 5, 2, 0, 6, 0, 0, 3, 0, 0, 0, 0,
#        1, 0, 0, 2, 0, 8, 0, 9, 0, 0, 2, 7, 4, 6, 0]

# numbers=[2, 1, 5, 0, 6, 0, 9, 0, 0, 7, 0, 0, 0, 0, 9, 1, 0, 0, 4, 0, 9, 3, 1, 0, 0, 5, 8, 0, 0, 1, 0, 0, 5, 0, 4, 0, 9, 0, 4, 0, 3, 0, 8, 0, 5, 0, 5, 0, 2, 0, 0, 6, 0, 0, 3, 8, 0, 0, 4, 1, 5, 0, 6, 0, 0, 6, 7, 0, 0, 0, 0, 2, 0, 0, 7, 0, 8, 0, 0, 1, 9]

# amar = []
# for i in range(9):
#     ligne = numbers1[i*9 : (i+1)*9]
#     amar.append(ligne)

# grid1 = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
#             [6, 0, 0, 1, 9, 5, 0, 0, 0],
#             [0, 9, 8, 0, 0, 0, 0, 6, 0],
#             [8, 0, 0, 0, 6, 0, 0, 0, 3],
#             [4, 0, 0, 8, 0, 3, 0, 0, 1],
#             [7, 0, 0, 0, 2, 0, 0, 0, 6],
#             [0, 6, 0, 0, 0, 0, 2, 8, 0],
#             [0, 0, 0, 4, 1, 9, 0, 0, 5],
#             [0, 0, 0, 0, 8, 0, 0, 7, 9]]

# grid= [ [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 7, 0, 0, 0, 0],
#         [0, 0, 0, 0, 7, 0, 0, 7, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 7, 0, 0, 0, 0]]

# dure= [[0, 3, 0, 0, 0, 0, 0, 0, 0], 
#         [0, 0, 0, 1, 9, 5, 0, 0, 0], 
#         [0, 9, 8, 0, 0, 0, 0, 6, 0], 
#         [8, 0, 0, 0, 6, 0, 0, 0, 0], 
#         [4, 0, 0, 0, 0, 3, 0, 0, 1], 
#         [0, 0, 0, 0, 2, 0, 0, 0, 0], 
#         [0, 6, 0, 0, 0, 0, 2, 8, 0], 
#         [0, 0, 0, 4, 1, 9, 0, 0, 5], 
#         [0, 0, 0, 0, 0, 0, 0, 7, 0]]

# dure2=[
#     [7, 0, 9, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 0, 0, 0, 0, 3, 5], 
#     [0, 0, 0, 0, 0, 0, 0, 0, 0], 
#     [8, 0, 0, 0, 0, 0, 9, 0, 6], 
#     [0, 5, 0, 0, 0, 3, 0, 0, 0], 
#     [0, 0, 0, 0, 2, 0, 7, 0, 0], 
#     [4, 0, 0, 6, 9, 0, 0, 0, 0], 
#     [0, 3, 0, 0, 0, 0, 0, 8, 0], 
#     [0, 0, 0, 7, 0, 0, 0, 0, 0]]
# # dure2=60.429

# start_time = time.time()
# boardTest=dure2
# try:
#     yassa = solve(boardTest)
#     print_board(boardTest)
# except:
#     print('except')
#     pass
# end_time = time.time()
# execution_time = end_time - start_time
# print("Temps d'exécution:", execution_time, "secondes")

