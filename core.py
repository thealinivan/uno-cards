#core file
import cv2 as cv
import matplotlib.pyplot as plt
from image_processing import getCardData

cardColors = ["r", "g", "b", "y"]
cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]
cardGeneralWilds = ["x", "y", "z"]

#run image recognition by opening a live stream using connected camera at index 0 
def liveStream(h, w):
    #setup
    vc = cv.VideoCapture(0)
    vc.set(3, 960)
    vc.set(4, 1280)
    currentCard = "r1"
    while vc.isOpened():
        rval, frame = vc.read()
        
        #...
        
        print("Current Card: " + str())
        cv.imshow("stream", frame)
        key = cv.waitKey(1)
        if key == 27: break
    cv.destroyWindow("stream")
    vc.release()

#run image recognition by reading images from files
def readFromFile():
    images = glob.glob('img/*.jpg')
    for name in images:
        img = cv.imread(name)
        frame = getCardData(img)
        #cv.imshow('file_read', frame)
        print(plt.imshow(i/255))
        plt.pause(0.001)

        
        

   