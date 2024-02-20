import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sudukoSolver
# import sudukoSolver2
from utils import showImage,preProcess, biggestContour, reorder,write_text_next_to_points,splitBoxes,getPredectionOneImage,imageBlank,getAllPreditions,displayNumbers,addGridImage,displayMultipleImages
import signal

timeToResolve = 5

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    print('signum',signum,frame)
    raise TimeoutException("La fonction a pris trop de temps pour s'exécuter.")


heightImg = 450
widthImg = 450
model = load_model('./model/model_trained.keras')
flag=False

cap = cv2.VideoCapture(1)
# Récupérer la largeur et la hauteur de la vidéo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Largeur de la vidéo:", width)
print("Hauteur de la vidéo:", height)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # img = cv2.resize(img, (widthImg, heightImg))
    imgThreshold = preProcess(img)
    
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES

    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    
    cv2.imshow('imgThreshold', imgThreshold)

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
        # cv2.imshow('imgBigContour', imgBigContour)
        cv2.imshow('imgWarpColored', imgWarpColored)
            
        if flag==False:
            # print('flag False')
            boxes = splitBoxes(imgWarpColored)
            numbers =getAllPreditions(boxes,model)
            # image_numbers = displayNumbers(imageBlank(),numbers, (255,255,255))
            # cv2.imshow('image_numbers', image_numbers)
            numbers=np.asarray(numbers)
            posArray = np.where(numbers>0 , 0 , 1)
            board = np.array_split(numbers,9)
            
            print('******** 1 ************')
            try:
                # Définir un gestionnaire de timeout pour SIGALRM
                signal.signal(signal.SIGALRM, timeout_handler)
                # Définir une alarme pour se déclencher après 5 secondes
                signal.alarm(timeToResolve)

                flag = sudukoSolver.solve(board)
                if flag:
                    print('flag',flag)
                    sudukoSolver.print_board(board)
            except TimeoutException:
                print('La fonction a pris trop de temps pour répondre')
                pass
            except Exception as e:
                print('Une erreur est survenue :', e)
                pass
            finally:
                # Annuler l'alarme après l'exécution de la fonction
                signal.alarm(0)
                pass
        if flag==True:
            # print("********** flag True ***********")
            flatList = [item for sublist in board for item in sublist]
            solvedNumbers =flatList*posArray
            imgSolvedDigits=displayNumbers(imageBlank(),solvedNumbers, (124,200,124))
            pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
            imgInvWarpColored = img.copy()
            imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (width, height))
            inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
            cv2.imshow('inv_perspective', inv_perspective)
        # else:
            # print('else flag')
            # cv2.imshow("my-img", img)

    else:
        # print('chanegment')
        # cv2.imshow("my-image", img)
        flag=False       


        

    # Display the resulting frame
    # cv2.imshow('imgContours', imgContours)
    # cv2.imshow('imgThreshold', imgThreshold)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
