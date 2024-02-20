import matplotlib.pyplot as plt
import cv2
import numpy as np

def showImage(img, name='my-image'):
    plt.imshow(img,cmap='gray')
    plt.title(name)
    plt.axis('off')
    plt.show()



def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0, cv2.BORDER_DEFAULT)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold



def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 30000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area



#### 3 - Reorder points for Warp Perspective
def reorder(myPoints):
   
    myPoints = myPoints.reshape((4, 2))

    #print(myPoints.shape)
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    #print('*****')
    #print(myPointsNew)
    
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def write_text_next_to_points(image, points, text_offset=10, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 0, 0), thickness=2):
    # Copier l'image pour éviter de modifier l'original
    image_with_text = image.copy()
    points = reorder(points)
    # Parcourir les points et écrire un texte à côté de chaque point
    for i, point in enumerate(points, start=1):
        # Coordonnées du point
        x, y = point[0]
        # Texte à écrire à côté du point
        text = f"{i}"
        # Dessiner le texte à côté du point
        cv2.putText(image_with_text, text, (x + text_offset, y + text_offset), font, font_scale, color, thickness)
    return image_with_text

#### 4 - TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes



#### 4 - GET PREDECTIONS ON ALL IMAGES
def getPredectionOneImage(image, model, show=True):
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
    minAccept=0.50
    if probVal> minAccept:
        cv2.putText(imageWithoutBorder,str(classIndex) + "   "+str(probVal), (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    if show== True :
        showImage(imageWithoutBorder)
    print(classIndex,probVal)
    return classIndex[0] if probVal > minAccept else 0


def getAllPreditions(boxes,model):
    result=[]
    for image in boxes:
        result.append(getPredectionOneImage(image,model, False))
    return result

#### 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img,numbers,color = (0,0,0)):

    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            currentBox=(y*9)+x
            if numbers[currentBox] != 0 :
                 cv2.putText(img, str(numbers[currentBox]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img



def imageBlank(heightImg=450, widthImg=450):
    return np.zeros((heightImg, widthImg, 3), np.uint8)


def addGridImage(image):
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



# Afficher les deux images
def displayMultipleImages(images):
    num_images = len(images)
    num_cols = 2
    num_rows = (num_images + 1) // num_cols  # Division entière pour obtenir le nombre de lignes nécessaires

    plt.figure(figsize=(10, 5))  # Taille de la figure

    for i, image in enumerate(images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Image {}".format(i + 1))
        plt.axis('off')

    plt.show()