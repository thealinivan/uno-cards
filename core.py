#core file
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image_processing import *
from keras.preprocessing.image import load_img as load 


#live stream / args: int - camera index
def liveStream(camIndex):
    print("Opening camera with index " + str(camIndex) + "...")
    vc = cv.VideoCapture(camIndex)
    vc.set(3, 800)
    vc.set(4, 600)
    currentCard = ""
    while vc.isOpened():
        rval, frame = vc.read()
        frame = np.asarray(frame)
        frame, detectedCard = getCardData(frame)
        cv.imshow("stream", frame)
        key = cv.waitKey(1)
        if key == 27: break
    cv.destroyWindow("stream")
    vc.release()

#form file / args: string - image name
def readFromFile(imgName):
    if os.path.isfile("img/" + imgName + ".jpg"):
        frame = np.asarray(load('img/'+ imgName +'.jpg', target_size=(600, 800)))
        frame = np.asarray(frame)
        print("Card Info: " + str(imgName))
        enhancedFrame, detectedCard = getCardData(frame)
        print(plt.imshow(frame))
        print("Detected as: " + str(detectedCard))
        plt.pause(0.001)
    

        

   