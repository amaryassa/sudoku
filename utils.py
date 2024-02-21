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
    imgBlur = cv2.GaussianBlur(imgGray, (9, 9), 0, cv2.BORDER_DEFAULT)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    
     # invert it so the grid lines and text are white
    # inverted = cv2.bitwise_not(imgThreshold, 0)

    # get a rectangle kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # morph it to remove some noise like random dots
    morph = cv2.morphologyEx(imgThreshold, cv2.MORPH_OPEN, kernel)

    # dilate to increase border size
    result = cv2.dilate(morph, kernel, iterations=1)
    return result



def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    bigCountour=None
    for i in contours:
        area = cv2.contourArea(i)
        if area > 15000:
            peri = cv2.arcLength(i, closed=True)
            approx = cv2.approxPolyDP(i, 0.01 * peri, closed=True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
                bigCountour=i
    return biggest,max_area,bigCountour




#### 3 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
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
        if type(image) == int:
            result.append(0)
        else:
            image =cv2.bitwise_not(image)
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
        plt.title("{}".format(i + 1))
        plt.axis('off')

    plt.show()
    


def grid_line_helper(img, shape_location, length=10):
    clone = img.copy()
    # if its horizontal lines then it is shape_location 1, for vertical it is 0
    row_or_col = clone.shape[shape_location]
    # find out the distance the lines are placed
    size = row_or_col // length

    # find out an appropriate kernel
    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    # erode and dilate the lines
    clone = cv2.erode(clone, kernel)
    clone = cv2.dilate(clone, kernel)

    return clone
    
def get_grid_lines(img, length=10):
    horizontal = grid_line_helper(img, 1, length)
    vertical = grid_line_helper(img, 0, length)
    return vertical, horizontal


def create_grid_mask(vertical, horizontal):
    # combine the vertical and horizontal lines to make a grid
    grid = cv2.add(horizontal, vertical)
    # threshold and dilate the grid to cover more area
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # find the list of where the lines are, this is an array of (rho, theta in radians)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    lines = draw_lines(grid, pts)
    # extract the lines so only the numbers remain
    mask = cv2.bitwise_not(lines)
    return mask


def draw_lines(img, lines):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    clone = img.copy()
    lines = np.squeeze(lines)

    for rho, theta in lines:
        # find out where the line stretches to and draw them
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return clone

def split_into_squares(warped_img):
    squares = []

    width = warped_img.shape[0] // 9

    # find each square assuming they are of the same side
    for j in range(9):
        for i in range(9):
            p1 = (i * width, j * width)  # Top left corner of a bounding box
            p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box
            squares.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])

    return squares

def clean_squares(squares):
    cleaned_squares = []
    i = 0

    for square in squares:
        new_img, is_number = clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
            i += 1

        else:
            cleaned_squares.append(0)

    return cleaned_squares

def clean_helper(img):
    # print(np.isclose(img, 0).sum())
    if np.isclose(img, 0).sum() / (img.shape[0] * img.shape[1]) >= 0.95:
        return np.zeros_like(img), False

    # if there is very little white in the region around the center, this means we got an edge accidently
    height, width = img.shape
    mid = width // 2
    if np.isclose(img[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (2 * width * 0.4 * height) >= 0.90:
        return np.zeros_like(img), False

    # center image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    start_x = (width - w) // 2
    start_y = (height - h) // 2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]

    return new_img, True