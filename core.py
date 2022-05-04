#core file
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image_processing import *

cardColors = ["r", "g", "b", "y"]
cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]

#run image recognition by opening a live stream using connected camera at index 0 
def liveStream(camIndex):
    print("Opening camera with index " + str(camIndex) + "...")
    vc = cv.VideoCapture(camIndex)
    vc.set(3, 960)
    vc.set(4, 1280)
    currentCard = ""
    while vc.isOpened():
        rval, frame = vc.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        enhancedFrame, detectedCard = getCardData(0, frame, cardColors, cardNumbers)
        print("Current Card: " + str(detectedCard))
        cv.imshow("stream", cv.cvtColor(enhancedFrame, cv.COLOR_BGR2RGB))
        key = cv.waitKey(1)
        if key == 27: break
    cv.destroyWindow("stream")
    vc.release()

#run image recognition by reading image from file
def readFromFile(imgName):
    if os.path.isfile("img/" + imgName + ".jpg"):
        frame = cv.cvtColor(cv.imread("img/" + imgName + ".jpg"), cv.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        print("Read from file: " + str(imgName))
        enhancedFrame, detectedCard = getCardData(1, frame, cardColors, cardNumbers)
        print(plt.imshow(enhancedFrame))
        print("Detected information: " + str(detectedCard))
        plt.pause(0.001)
    

        
        

   