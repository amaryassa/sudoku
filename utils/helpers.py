import matplotlib.pyplot as plt
import cv2
import numpy as np

def preProcess(img):
    # Convertir l'image en niveaux de gris
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Appliquer un flou gaussien pour réduire le bruit
    imgBlur = cv2.GaussianBlur(imgGray, (9, 9), 0, cv2.BORDER_DEFAULT)
    # Appliquer un seuillage adaptatif pour obtenir une image binaire inversée
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Définir un élément structural pour les opérations morphologiques
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Appliquer une opération morphologique d'ouverture pour éliminer le bruit
    morph = cv2.morphologyEx(imgThreshold, cv2.MORPH_OPEN, kernel)
    # Appliquer une dilatation pour augmenter la taille des bordures
    result = cv2.dilate(morph, kernel, iterations=1)
    return result

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    # contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    return  contours, hierarchy 
    

def findBiggestContour(contours):
    """
    Trouve le plus grand contour parmi une liste de contours.

    Args:
    contours: Liste des contours à analyser.

    Returns:
    biggest: Les coordonnées des points du plus grand contour.
    max_area: L'aire du plus grand contour.
    bigContour: Le plus grand contour lui-même.
    """
    biggest = np.array([])
    max_area = 0
    bigContour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 15000:
            peri = cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, closed=True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
                bigContour = contour
    return biggest, max_area, bigContour

def reorderPointsForWarp(myPoints):
    # Remodeler les points pour obtenir une matrice de forme (4, 2)
    myPoints = myPoints.reshape((4, 2))
    # Initialiser une nouvelle matrice pour les points réordonnés
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    # Calculer la somme des coordonnées des points selon l'axe 1 (somme des x et des y)
    add = myPoints.sum(1)
    # Trouver l'indice du point avec la somme minimale et l'ajouter en premier dans la nouvelle matrice
    myPointsNew[0] = myPoints[np.argmin(add)]
    # Trouver l'indice du point avec la somme maximale et l'ajouter en dernier dans la nouvelle matrice
    myPointsNew[3] = myPoints[np.argmax(add)]
    # Calculer la différence entre les coordonnées des points selon l'axe 1 (différence des x et des y)
    diff = np.diff(myPoints, axis=1)
    # Trouver l'indice du point avec la différence minimale et l'ajouter en deuxième dans la nouvelle matrice
    myPointsNew[1] = myPoints[np.argmin(diff)]
    # Trouver l'indice du point avec la différence maximale et l'ajouter en troisième dans la nouvelle matrice
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # Retourner les points réordonnés pour la perspective en warp
    return myPointsNew

def drawContours(img, contours, color=(0, 255, 0), size=5):
    imageWithCountour = img.copy()
    cv2.drawContours(imageWithCountour, contours, -1, color, size)
    return imageWithCountour

def showImage(img, name='my-image'):
    plt.imshow(img,cmap='gray')
    plt.title(name)
    plt.axis('off')
    plt.show()

def showMultipleImages(images,num_cols=2):
    # Calculer le nombre total d'images à afficher
    num_images = len(images)
    
    # Calculer le nombre de lignes nécessaires pour afficher toutes les images
    num_rows = (num_images + 1) // num_cols  # Division entière pour obtenir le nombre de lignes nécessaires

    # Définir la taille de la figure à afficher
    plt.figure(figsize=(12, 8))  # Taille de la figure

    # Parcourir chaque image et l'afficher dans une sous-figure
    for i, image in enumerate(images):
        if type(image) != int:
            # Ajouter une sous-figure pour l'image actuelle
            plt.subplot(num_rows, num_cols, i + 1)
            
            # Afficher l'image en convertissant les couleurs de BGR à RGB
            plt.imshow(image,cmap='gray')
            
            # Définir le titre de l'image comme son numéro d'index
            plt.title("{}".format(i + 1))
            
            # Masquer les axes
            plt.axis('off')

    # Afficher la figure contenant toutes les images
    plt.show()
    
def imageBlank(heightImg, widthImg):
    #def imageBlank(heightImg=450, widthImg=450):
    return np.zeros((heightImg, widthImg, 3), np.uint8)

def drawGrid(image):
    """
    Ajoute une grille à une image donnée.

    Args:
    image: L'image à laquelle ajouter la grille.

    Returns:
    L'image avec une grille ajoutée.
    """
    img= image.copy()
    secW = int(img.shape[1])
    secH = int(img.shape[0])
    for i in range(0, 10):
        thickness = 4 if i % 3 == 0 else 2  # Épaisseur 4 pour les lignes 3x3, sinon 2
        # Lignes verticales
        cv2.line(img, (i * secW // 9, 0), (i * secW // 9, secH), (255, 0, 0), thickness)
        # Lignes horizontales
        cv2.line(img, (0, i * secH // 9), (secW, i * secH // 9), (255, 0, 0), thickness)
    return img

def split_into_squares(warped_img):
    squares = []  # Initialisation d'une liste pour stocker les carrés découpés

    width = warped_img.shape[0] // 9  # Largeur d'un carré, en supposant une grille 9x9

    # Parcours de chaque ligne et colonne pour découper l'image en carrés
    for j in range(9):  # Parcours des lignes
        for i in range(9):  # Parcours des colonnes
            # Détermination des coins du carré actuel
            p1 = (i * width, j * width)  # Coin supérieur gauche du carré
            p2 = ((i + 1) * width, (j + 1) * width)  # Coin inférieur droit du carré
            
            # Extraction du carré de l'image en utilisant les coins déterminés
            square = warped_img[p1[1]:p2[1], p1[0]:p2[0]]
            
            # Ajout du carré découpé à la liste des carrés
            squares.append(square)

    return squares  # Retour de la liste des carrés découpés

def clean_and_center_image(img,i=0):
    # Vérifier si la proportion de pixels noirs dans l'image est supérieure à 95%
    if np.isclose(img, 0).sum() / (img.shape[0] * img.shape[1]) >= 0.95:
        # Si oui, retourner une image entièrement noire et False (aucun chiffre trouvé)
        return np.zeros_like(img), False

    # Vérifier si la région autour du centre de l'image contient très peu de pixels blancs
    height, width = img.shape
    mid = width // 2
    if np.isclose(img[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (2 * width * 0.4 * height) >= 0.92: #0.9
        # Si oui, retourner une image entièrement noire et False (aucun chiffre trouvé)
        return np.zeros_like(img), False

    # Centrer l'image en fonction du plus grand contour externe trouvé
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    start_x = (width - w) // 2
    start_y = (height - h) // 2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]

    # Retourner l'image centrée et True (un chiffre a été trouvé)
    return new_img, True

def clean_squares(squares):
    cleaned_squares = []  # Initialisation d'une liste pour stocker les carrés nettoyés

    # Parcours de chaque carré dans la liste fournie
    for square in squares:
        # Nettoyage du carré et vérification s'il contient un chiffre
        new_img, is_number = clean_and_center_image(square)

        # Si le carré contient un chiffre
        if is_number:
            # Ajout du carré nettoyé à la liste des carrés nettoyés
            cleaned_squares.append(new_img)

        # Si le carré ne contient pas de chiffre
        else:
            # Ajout de zéro à la liste des carrés nettoyés (pour maintenir l'alignement avec l'index original)
            cleaned_squares.append(0)

    # Retour de la liste des carrés nettoyés
    return cleaned_squares
  
def grid_line_helper(img, shape_location, length=10):
    clone = img.copy()
    # si ce sont des lignes horizontales, shape_location vaut 1, pour les verticales c'est 0
    row_or_col = clone.shape[shape_location]
    # déterminer la distance entre les lignes
    size = row_or_col // length

    # trouver un noyau approprié
    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    # éroder et dilater les lignes
    clone = cv2.erode(clone, kernel)
    clone = cv2.dilate(clone, kernel)

    return clone
    
def get_grid_lines(img, length=10):
    horizontal = grid_line_helper(img, 1, length)
    vertical = grid_line_helper(img, 0, length)
    return vertical, horizontal

def create_grid_mask(vertical, horizontal):
    # Combiner les lignes verticales et horizontales pour créer une grille
    grid = cv2.add(horizontal, vertical)
    # Appliquer un seuillage adaptatif pour binariser l'image
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    # Dilater la grille pour couvrir une plus grande surface
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # Détecter les lignes de la grille à l'aide de la transformée de Hough
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)
    # Dessiner les lignes détectées sur une image vierge
    lines = draw_lines(grid, pts)
    # Inverser les couleurs pour obtenir un masque de grille
    mask = cv2.bitwise_not(lines)
    return mask

def create_grid_mask2(vertical, horizontal):
    # Combiner les lignes verticales et horizontales pour créer une grille
    grid = cv2.add(horizontal, vertical)
    # Appliquer un seuillage adaptatif pour binariser l'image
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    # Dilater la grille pour couvrir une plus grande surface
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # Détecter les lignes de la grille à l'aide de la transformée de Hough
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)
    # Dessiner les lignes détectées sur une image vierge
    lines = draw_lines(grid, pts)
    # Inverser les couleurs pour obtenir un masque de grille
    # mask = cv2.bitwise_not(lines)
    return lines

def draw_lines(img, lines):
    clone = img.copy()  # Création d'une copie de l'image d'origine
    lines = np.squeeze(lines)  # Suppression des dimensions superflues dans les lignes

    for rho, theta in lines:  # Parcours de chaque ligne détectée par la transformée de Hough
        # Calcul des coordonnées des points de départ et d'arrivée de la ligne
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        
        # Dessin de la ligne sur l'image clone avec une épaisseur de 2 pixels
        cv2.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    return clone  # Retour de l'image avec les lignes dessinées

def drawSudokuDigits(img, numbers, color=(0, 0, 0)):
    # Calculer la largeur et la hauteur de chaque section de la grille
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    
    # Parcourir chaque case de la grille
    for x in range(0, 9):
        for y in range(0, 9):
            # Calculer l'index de la case actuelle
            currentBox = (y * 9) + x
            
            # Vérifier si la case contient un nombre différent de zéro
            if numbers[currentBox] != 0:
                # Dessiner le nombre au centre de la case
                cv2.putText(img, str(numbers[currentBox]),
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

def getPredectionOneImage(image, model, show=True):
    # transformer en fond blanc et texte en noir
    image =cv2.bitwise_not(image)
    img = np.asarray(image)
    border=4
    img = img[border:img.shape[0] - border, border:img.shape[1] -border]
    img = cv2.resize(img,(32,32))
    img = img/255
    imageWithoutBorder = img.copy() 
    img = img.reshape(1,32,32,1)
    #### PREDICT
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)
    probVal= np.amax(predictions)
    minAccept=0.90
    if probVal> minAccept:
        cv2.putText(imageWithoutBorder,str(classIndex) + "   "+str(probVal), (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    if show== True :
        showImage(image)
    print(classIndex,probVal)
    return classIndex[0] if probVal > minAccept else 0

def getAllPreditions(boxes,model):
    result=[]
    for image in boxes:
        if type(image) == int:
            result.append(0)
        else:
            result.append(getPredectionOneImage(image,model, False))
    return result


# Gestionnaire pour gérer le dépassement de temps
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("La fonction a pris trop de temps pour s'exécuter.")