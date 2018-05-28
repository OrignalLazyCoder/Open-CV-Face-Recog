import os
import cv2
import numpy as np
from PIL import Image

recogniser = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

def getImagesAndIds(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg , 'uint8')
        ID =int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        Ids.append(ID)
        cv2.imshow("training" , faceNp)
        cv2.waitKey(10)
    return np.array(Ids), faces

Ids , faces= getImagesAndIds(path)
recogniser.train(faces ,Ids)
recogniser.save('recogniser/trainingData.yml')
cv2.destroyAllWindows()
