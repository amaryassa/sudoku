import signal
import time

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("La fonction a pris trop de temps pour s'exécuter.")

def print_board(board):
    """
    Affiche le tableau du Sudoku.

    Args:
        board (list[list[int]]): Un tableau de Sudoku 9x9 représenté sous forme d'une liste de listes d'entiers.

    Returns:
        None.
    """

    boardString = ""
    for i in range(9):
        for j in range(9):
            boardString += str(board[i][j]) + " "
            if (j + 1) % 3 == 0 and j != 0 and j + 1 != 9:
                boardString += "| "

            if j == 8:
                boardString += "\n"

            if j == 8 and (i + 1) % 3 == 0 and i + 1 != 9:
                boardString += "- - - - - - - - - - - \n"
    print(boardString)


def find_empty(board):
    """
    Trouve une case vide dans le tableau du Sudoku.

    Args:
        board (list[list[int]]): Un tableau de Sudoku 9x9 représenté sous forme d'une liste de listes d'entiers.

    Returns:
        tuple[int, int]|None: La position de la première case vide trouvée sous forme d'un tuple d'indices de ligne et de colonne, ou None si aucune case vide n'est trouvée.
    """

    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None


def valid(board, pos, num):
    """
    Vérifie si un numéro est valide dans une case du tableau du Sudoku.

    Args:
        board (list[list[int]]): Un tableau de Sudoku 9x9 représenté sous forme d'une liste de listes d'entiers.
        pos (tuple[int, int]): La position de la case à vérifier sous forme d'un tuple d'indices de ligne et de colonne.
        num (int): Le numéro à vérifier.

    Returns:
        bool: True si le numéro est valide dans la case, False sinon.
    """

    for i in range(9):
        if board[i][pos[1]] == num:
            return False

    for j in range(9):
        if board[pos[0]][j] == num:
            return False

    start_i = pos[0] - pos[0] % 3
    start_j = pos[1] - pos[1] % 3
    for i in range(3):
        for j in range(3):
            if board[start_i + i][start_j + j] == num:
                return False
    return True


def solve(board):
    """
    Résout le tableau du Sudoku en utilisant l'algorithme de backtracking.

    Args:
        board (list[list[int]]): Un tableau de Sudoku 9x9 représenté sous forme d'une liste de listes d'entiers.

    Returns:
        bool: True si le tableau du Sudoku est résolu, False sinon.
    """
    empty = find_empty(board)
    if not empty:
        return True

    for nums in range(1, 10):
        if valid(board, empty, nums):
            board[empty[0]][empty[1]] = nums

            # Définir un gestionnaire de timeout pour SIGALRM
            signal.signal(signal.SIGALRM, timeout_handler)
            # Définir une alarme pour se déclencher après 30 secondes
            signal.alarm(5)

            try:
                
                if solve(board):  # appel récursif
                    return True
            except TimeoutException:
                return False
            finally:
                # Annuler l'alarme après l'exécution de la fonction
                signal.alarm(0)

            board[empty[0]][empty[1]] = 0  # Remettre cette case à 0, car ce nombre est incorrect
    return False




# if __name__ == "__main__":
#     board = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
#         [6, 0, 0, 1, 9, 5, 0, 0, 0],
#         [0, 9, 8, 0, 0, 0, 0, 6, 0],
#         [8, 0, 0, 0, 6, 0, 0, 0, 3],
#         [4, 0, 0, 8, 0, 3, 0, 0, 1],
#         [7, 0, 0, 0, 2, 0, 0, 0, 6],
#         [0, 6, 0, 0, 0, 0, 2, 8, 0],
#         [0, 0, 0, 4, 1, 9, 0, 0, 5],
#         [0, 0, 0, 0, 8, 0, 0, 7, 9]]
    
#     bloque= [ [0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 7, 0, 0, 0, 0],
#             [0, 0, 0, 0, 7, 0, 0, 7, 0],
#             [0, 0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 7, 0, 0, 0, 0]]
    
#     dure= [[0, 3, 0, 0, 0, 0, 0, 0, 0], 
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
#     # dure2= 87.22199702262878 secondes
# toTest=dure2
# print(toTest)
# print_board(toTest)
# start_time = time.time()
# solve(toTest)
# print_board(toTest)
# end_time = time.time()
# execution_time = end_time - start_time
# print("Temps d'exécution:", execution_time, "secondes")
