import cv2
import numpy as np
import tensorflow as tf
# import sudoku_solver2 as ss
from time import sleep
import operator
from tensorflow.keras.models import load_model
import sudukoSolver
from utils import showImage,preProcess, biggestContour, reorder,write_text_next_to_points,splitBoxes,getPredectionOneImage,imageBlank,getAllPreditions,displayNumbers,addGridImage,displayMultipleImages


marge=4
case=28+2*marge
taille_grille=9*case
flag=0
cap=cv2.VideoCapture(0)
model = load_model('./model/model_trained.keras')

maxArea=0
while True:
    ret, frame=cap.read()
    if maxArea==0:
        cv2.imshow("frame", frame)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (5, 5), 0)
    thresh=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_grille=None
    maxArea=0
    for c in contours:
        area=cv2.contourArea(c)
        if area>25000:
            peri=cv2.arcLength(c, True)
            polygone=cv2.approxPolyDP(c, 0.02*peri, True)
            if area>maxArea and len(polygone)==4:
                contour_grille=polygone
                maxArea=area
    if contour_grille is not None:
        points=np.vstack(contour_grille).squeeze()
        points=sorted(points, key=operator.itemgetter(1))
        if points[0][0]<points[1][0]:
            if points[3][0]<points[2][0]:
                pts1=np.float32([points[0], points[1], points[3], points[2]])
            else:
                pts1=np.float32([points[0], points[1], points[2], points[3]])
        else:
            if points[3][0]<points[2][0]:
                pts1=np.float32([points[1], points[0], points[3], points[2]])
            else:
                pts1=np.float32([points[1], points[0], points[2], points[3]])
        pts2=np.float32([[0, 0], [taille_grille, 0], [0, taille_grille], [taille_grille, taille_grille]])
        M=cv2.getPerspectiveTransform(pts1, pts2)
        grille=cv2.warpPerspective(frame, M, (taille_grille, taille_grille))
        grille=cv2.cvtColor(grille, cv2.COLOR_BGR2GRAY)
        grille=cv2.adaptiveThreshold(grille, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
        cv2.imshow("grille", grille)
        if flag==0:
            print('toto')
            boxes = splitBoxes(grille)
            numbers =getAllPreditions(boxes,model)
            numbers=np.asarray(numbers)
            posArray = np.where(numbers>0 , 0 , 1)
            board = np.array_split(numbers,9)
            result=sudukoSolver.solve(board)
        #result=None
        if result is not None:
            flag=1
            flatList = [item for sublist in board for item in sublist]
            solvedNumbers =flatList*posArray
            imgSolvedDigits=displayNumbers(imageBlank(),solvedNumbers, (124,200,124))
            
            M=cv2.getPerspectiveTransform(pts2, pts1)
            h, w, c=frame.shape
            fondP=cv2.warpPerspective(imgSolvedDigits, M, (w, h))
            img2gray=cv2.cvtColor(fondP, cv2.COLOR_BGR2GRAY)
            ret, mask=cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask=mask.astype('uint8')
            mask_inv=cv2.bitwise_not(mask)
            img1_bg=cv2.bitwise_and(frame, frame, mask=mask_inv)
            img2_fg=cv2.bitwise_and(fondP, fondP, mask=mask).astype('uint8')
            dst=cv2.add(img1_bg, img2_fg)
            cv2.imshow("frame", dst)
        else:
            cv2.imshow("frame", frame)
    else:
        flag=0
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
