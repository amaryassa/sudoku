import cv2
import numpy as np
from tensorflow.keras.models import load_model
from libs import sudukoSolver
from utils import helpers
import signal


# Initialisation de la taille des images
heightImg = 360
widthImg = 360

# Initialisation de la taille de la capture vidéo
width = 800
height = 600
# width = 960
# height = 720

# Capture vidéo
cap = cv2.VideoCapture(0)
cap.set(3, width)  # Largeur
cap.set(4, height)  # Hauteur
cap.set(10, 150)  # Luminosité

# Durée maximale avant de considérer la grille comme non résolue
timeToResolve = 5
# Dictionnaire pour stocker les grilles résolues
seen = dict()
seenAlready=''
# Indicateur pour le temps écoulé
flag = False
# Chargement du modèle Keras pour la prédiction des chiffres
model = load_model('./model/model_trained.keras')


# Vérification si la caméra est ouverte
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()
    
while True:
    ret, img = cap.read()
    if not ret:
        print("Impossible de recevoir le flux vidéo. Sortie...")
        break
    
    # Redimensionner l'image capturée pour une meilleure visualisation
    img = cv2.resize(img, (width, height))
    img_result = img.copy()
    
    # Prétraitement de l'image pour obtenir la grille de Sudoku
    imgThreshold = helpers.preProcess(img)
    #cv2.imshow('imgThreshold', imgThreshold)
    
    # Recherche des contours dans l'image
    contours, hierarchy = helpers.getContours(imgThreshold)
    
    # Recherche du plus grand contour (grille de Sudoku)
    biggest, maxArea, countour = helpers.findBiggestContour(contours)
    allContours = helpers.drawContours(img, contours, (0, 255, 0),2)
    if biggest.size != 0:
        # Réorganisation des points pour la perspective
        biggest = helpers.reorderPointsForWarp(biggest)
        allContours = helpers.drawContours(allContours, biggest, (255, 0, 255), 10)
        allContours = helpers.drawContours(allContours, [countour],(0, 0, 255),3)

        # Transformation de perspective pour obtenir une image de grille de Sudoku
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        cv2.imshow('imgWarpColored', imgWarpColored)
        # Prétraitement de l'image de la grille
        imgWarpProcessed = helpers.preProcess(imgWarpColored)

        # Obtention des lignes de la grille
        vertical_lines, horizontal_lines = helpers.get_grid_lines(imgWarpProcessed)

        # Création du masque pour la grille
        maskGrid = helpers.create_grid_mask(vertical_lines, horizontal_lines)

        # Extraction des chiffres de la grille
        numbersImages = cv2.bitwise_and(imgWarpProcessed, maskGrid)
        
        squares = helpers.split_into_squares(numbersImages)
        squares_processed = helpers.clean_squares(squares)
       
        # Obtention des prédictions pour chaque chiffre dans la grille
        if flag == False:
            numbers = helpers.getAllPreditions(squares_processed, model)
            print(numbers)
            squares_guesses = tuple(numbers)

            IA_numbers = helpers.drawSudokuDigits(helpers.imageBlank(heightImg,widthImg), numbers, (255,255,255))
            cv2.imshow("Prédictions de l'IA", IA_numbers)
            cv2.imshow("Grille originale", numbersImages)
            numbers = np.asarray(numbers)
            posArray = np.where(numbers > 0 , 0 , 1)
            board = np.array_split(numbers, 9)
         
            # Si cette grille a déjà été résolue, récupérer la solution
            if squares_guesses in seen:

                board = seen[squares_guesses]
                flag = True
                seenAlready="deja vu ! "
            else :
                seenAlready=''
                try:
                    # Définir un gestionnaire de timeout pour SIGALRM
                    signal.signal(signal.SIGALRM, helpers.timeout_handler)
                    # Définir une alarme pour se déclencher après 5 secondes
                    signal.alarm(timeToResolve)
                    # Résolution de la grille
                    flag = sudukoSolver.solve(board)
                    if flag:
                        sudukoSolver.print_board(board)
                        seen[squares_guesses] = board
                    else:
                        seen[squares_guesses] = False
                except helpers.TimeoutException:
                    print('La fonction a pris trop de temps pour répondre')
                    pass
                except Exception as e:
                    print('Une erreur est survenue :', e)
                    pass
                finally:
                    # Annuler l'alarme après l'exécution de la fonction
                    signal.alarm(0)
                    pass
        if flag == True:
            flatList = [item for sublist in board for item in sublist]
            solvedNumbers = flatList * posArray
            imgSolvedDigits = helpers.drawSudokuDigits(helpers.imageBlank(heightImg,widthImg), solvedNumbers, (124,200,124))
            pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
            imgInvWarpColored = img.copy()
            imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (width, height))
            inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
            img_result = inv_perspective
    else:
        flag = False
        seenAlready=''

    # Affichage de l'image résultante
    cv2.putText(img_result, seenAlready, (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('drawContours', allContours)

    cv2.imshow('window', img_result)
    if cv2.waitKey(1) == ord('q'):
        break

# Libération de la capture vidéo et fermeture des fenêtres
cap.release()
cv2.destroyAllWindows()
