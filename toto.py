import cv2
import numpy as np

def find_horizontal_lines(img):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer un filtre de Sobel pour détecter les contours verticaux
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)

    # Seuiller l'image pour obtenir des contours binaires
    _, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Appliquer une opération de dilatation pour fusionner les contours
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)

    # Trouver les contours des lignes horizontales
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les lignes horizontales sur une image vide
    lines_img = np.zeros_like(gray)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w / h > 5:
            cv2.drawContours(lines_img, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(lines_img, [cnt], 0, 0, -1)

    # Appliquer une opération de dilatation pour élargir les lignes
    close = cv2.morphologyEx(lines_img, cv2.MORPH_DILATE, None, iterations=2)

    return close

# Charger l'image
img = cv2.imread("images/1.png")

# Appeler la fonction pour trouver les lignes horizontales
horizontal_lines = find_horizontal_lines(img)

# Afficher l'image des lignes horizontales
cv2.imshow("Horizontal Lines", horizontal_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()