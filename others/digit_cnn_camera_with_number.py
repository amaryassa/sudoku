import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
#####################################

#### LOAD THE TRAINNED MODEL 
model = load_model('./model/model_trained.keras')
#


#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img



def showImage(img, name='my-img'):
    plt.imshow(img,cmap='gray')
    plt.title(name)
    #plt.axis('off')
    plt.show()


def predictImage(image, model):
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    img = img.reshape(1,32,32,1)
    #### PREDICT
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)
    probVal= np.amax(predictions)
    print(classIndex,probVal)
    if probVal> threshold:
        cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal), (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    showImage(imgOriginal)


# imgOriginal = cv2.imread("numbers_test/9.png")
# predictImage(imgOriginal,model)


#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)



while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    amar = img
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    amar = preProcessing(amar)
    cv2.imshow("Processsed Image",amar)
    img = img.reshape(1,32,32,1)
    #### PREDICT
    predictions = model.predict(img)
    classIndex = int(np.argmax(predictions, axis=1))

    #print(predictions)
    probVal= np.amax(predictions)
    print(classIndex,probVal)

    if probVal> threshold:
        cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break