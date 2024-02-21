import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sudukoSolver
# import sudukoSolver2
from utils import   get_grid_lines,showImage,preProcess, biggestContour, reorder,write_text_next_to_points,splitBoxes,getPredectionOneImage,imageBlank,getAllPreditions,displayNumbers,addGridImage,displayMultipleImages
import utils
import signal




timeToResolve = 5 # en secondes

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    print('signum',signum,frame)
    raise TimeoutException("La fonction a pris trop de temps pour s'exécuter.")


heightImg = 450
widthImg = 450
model = load_model('./model/model_trained.keras')
flag=False

cap = cv2.VideoCapture(0)
# Récupérer la largeur et la hauteur de la vidéo
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

width = 960
height = 720
# width is id number 3, height is id 4
cap.set(3, width)
cap.set(4, height)
# change brightness to 150
cap.set(10, 150)
seen = dict()
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
    img = cv2.resize(img, (width, height))
    img_result = img.copy()
    imgThreshold = preProcess(img)
    
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES

    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    # contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    
    
    

    # cv2.imshow('imgThreshold', imgThreshold)
    biggest, maxArea,countour = biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest = reorder(biggest)

        cv2.drawContours(imgBigContour, biggest, -1, (255, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
        cv2.drawContours(imgBigContour, [countour], -1, (255, 0, 0), 5) # DRAW THE BIGGEST CONTOUR
        # cv2.imshow('imgBigContour', imgBigContour)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        # imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8) 
        #lignes en blanc - chiffre en blanc - fond noir
        imgWarpProcessed = preProcess(imgWarpColored)
        cv2.imshow('imgWarpProcessed', imgWarpProcessed)

        
        vertical_lines, horizontal_lines = get_grid_lines(imgWarpProcessed)

        # les lignes en noinr -  fond blanc
        maskGrid =utils.create_grid_mask(vertical_lines,horizontal_lines)
        # cv2.imshow('maskGrid', maskGrid)
        #on va garder que les chifre en blanc
        numbersImages = cv2.bitwise_and(imgWarpProcessed, maskGrid)
        # cv2.imshow('numbers', numbers)
        squares=utils.split_into_squares(numbersImages)
        squares_processed = utils.clean_squares(squares)
       

            
        if flag==False:
            print('flag False')
            numbers =getAllPreditions(squares_processed,model)
            squares_guesses = tuple(numbers)


            image_numbers = displayNumbers(imageBlank(),numbers, (255,255,255))
            cv2.imshow('ce qui l IA a reconnu', image_numbers)
            numbers=np.asarray(numbers)
            posArray = np.where(numbers>0 , 0 , 1)
            board = np.array_split(numbers,9)
            
              # if squares_guesses in seen and seen[squares_guesses] is False:
            #     continue

            # if we already solved this puzzle, just fetch the solution
            if squares_guesses in seen:
                print('########## déjà vu #############')
                board= seen[squares_guesses]
                flag=True
            else :
                print('******** 1 ************')
                try:
                    # Définir un gestionnaire de timeout pour SIGALRM
                    signal.signal(signal.SIGALRM, timeout_handler)
                    # Définir une alarme pour se déclencher après 5 secondes
                    signal.alarm(timeToResolve)

                    flag = sudukoSolver.solve(board)
                    if flag:
                        sudukoSolver.print_board(board)
                        seen[squares_guesses]=board
                    else:
                        seen[squares_guesses]=False
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
            # cv2.imshow('inv_perspective', inv_perspective)
            img_result=inv_perspective
        # else:
            # print('else flag')
            # cv2.imshow("my-img", img)
    else:
        flag=False
        
        
    cv2.imshow('window', img_result)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
