#main file
import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from data_capturing import collectData
from image_processing import trainModels
from image_processing import getCardData
from keras.preprocessing.image import load_img as load 

cardColors = ["r", "g", "b", "y"]
cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]

#MENU
#menu
def getUserOption():
    print("")
    print ("-----------------")
    print ("*** Uno Cards ***")
    print ("-----------------")
    print("[1] Read from file")
    print("[2] Live camera stream")
    print("[3] Data capturing")
    print("[0] Exit")
    option = input("::")
    return option

#def read from file submenu
def getUserReadSubMenuOption():
    print("[1] Select card")
    print("[2] All data")
    print("[9] Return")
    option = input("::")
    return option

#camIndex
def getUserCamIndex():
    print("Please input CAMERA INDEX: 0, 1, 2 etc")
    camIndex = input("::")
    return int(camIndex)

#card selection
def getUserCard():
    print("PLease input COLOR: r,g,b,y")
    userCard = input("::")
    print("Please input NUMBER: 0,1,2,3,4,5,6,7,8,9,d,a,n")
    userCard += input("::")
    return userCard 

#CORE
#live stream / args: int - camera index
def liveStream(camIndex):
    print("Opening camera with index " + str(camIndex) + "...")
    vc = cv.VideoCapture(camIndex)
    vc.set(3, 800)
    vc.set(4, 600)
    i = 0
    while vc.isOpened():
        rval, frame = vc.read()
        if i > 20:
            frame = np.asarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            frame, detectedCard = getCardData(frame)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("frame", frame)
        i+=1
        key = cv.waitKey(1)
        if key == 27: break
    cv.destroyWindow("stream")
    vc.release()

#form file / args: string - image name
def readFromFile(imgName):
    if os.path.isfile("img/" + imgName + ".jpg"):
        frame = np.asarray(load('img/'+ imgName +'.jpg', target_size=(600,800)))
        print("Card Info: " + str(imgName))
        frame, detectedCard = getCardData(frame)
        print(plt.imshow(frame))
        print("Detected as: " + str(detectedCard))
        plt.pause(0.001)

#PROGRAM
#train ml models
print("Initializing ML models..")
trainModels()

#main loop
opt = getUserOption()
while opt != "0":
    if opt == "1": 
        readOpt = getUserReadSubMenuOption()
        while readOpt != "0":
            if readOpt == "1":  readFromFile(getUserCard())
            elif readOpt == "2": 
                for c in cardColors: [readFromFile(c+str(n)) for n in cardNumbers] 
            elif readOpt == "9": break
            elif readOpt == "0": sys.exit("Terminated") 
            else: print("Invalid option !")
            readOpt = getUserReadSubMenuOption()
    elif opt == "2": liveStream(getUserCamIndex())
    elif opt == "3": collectData(getUserCamIndex())
    elif opt == "0": sys.exit("Terminated") 
    else: print("Invalid option !")
    opt = getUserOption()
    







