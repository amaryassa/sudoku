import cv2

import numpy as np
from tensorflow.keras.models import load_model
import sudukoSolver
from utils import showImage,preProcess, biggestContour, reorder,write_text_next_to_points,splitBoxes,getPredectionOneImage,imageBlank,getAllPreditions,displayNumbers,addGridImage,displayMultipleImages


heightImg = 450
widthImg = 450
model = load_model('./model/model_trained.keras')
img=cv2.imread ("images/1.png")
img = cv2.resize(img, (widthImg, heightImg))

# showImage(img, 'original')

imgThreshold = preProcess(img)
#showImage(imgThreshold, 'original')


imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES

## #cv.RETR_LIST, cv.RETR_TREE, cv.RETR_CCOMP, cv.RETR_EXTERNAL
## CHAIN_APPROX_NONE CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1 CHAIN_APPROX_TC89_KCOS 

contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

#showImage(imgContours, 'imgContours')
#showImage(imgBigContour, 'imgBigContour')

biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
    biggest = reorder(biggest)
    # print('_________')
    #print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (255, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8) 
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

#showImage(write_text_next_to_points(imgBigContour, biggest), 'imgBigContour')
# showImage(imgWarpColored, 'imgWarpColored')


# all boxes (small images)
boxes = splitBoxes(imgWarpColored)

#getPredectionOneImage(boxes[78],model)

numbers =getAllPreditions(boxes,model)


image_numbers = displayNumbers(imageBlank(),numbers, (255,255,255))
# imageNumberWithGrid=addGridImage(image_numbers)


# pour qu'ensuite on remplis sur l'image original que les chiffres manquants, ceux qui ont un 1 ici

numbers=np.asarray(numbers)
posArray = np.where(numbers>0 , 0 , 1)
board = np.array_split(numbers,9)
# print(board)
try:
    sudukoSolver.solve(board)
except:
    pass

flatList = [item for sublist in board for item in sublist]
solvedNumbers =flatList*posArray


imgSolvedDigits=displayNumbers(imageBlank(),solvedNumbers, (124,200,124))
displayMultipleImages([imgWarpColored, addGridImage(imgSolvedDigits),addGridImage(image_numbers)])



pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
imgInvWarpColored = img.copy()
imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)


showImage(inv_perspective)