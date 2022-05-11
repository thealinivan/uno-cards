#image processing file
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img as load 
from sklearn.utils import Bunch
from augm import *
from time import sleep

cardColors = ["r", "g", "b", "y"]
colors = ["RED","GREEN", "BLUE", "YELLOW"]
cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]
colorClf = KNeighborsClassifier(2)
numberClf = KNeighborsClassifier(2)
resX = 600
resY = 800

#get contours / args: numpy array - image / return numpy array - contours
def getContour(frame):
    mono = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.blur(mono, (10, 10))
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 2)
    #HSVmin = np.array([0, 55, 55])
    #HSVmax = np.array([359, 255, 255])
    #HSVim = cv.cvtColor(frame, cv.COLOR_RGB2HSV);
    #th = cv.inRange(HSVim, HSVmin, HSVmax)
    kernel = np.ones((5, 5), np.uint8)
    close = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
    cont, t = cv.findContours(close, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = ()
    maxCnt = 1600
    for c in cont: 
        if len(c) > len(cnt) and len(c) < maxCnt: cnt = c
    return cnt

#get thresholded image / args: numpy array - image / return: numpy array - thresholded image
def getThreshold(frame):
    mono = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.blur(mono, (5, 5))
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 2)
    return th

#get crops / args: numpy array - contour / return: list - numpy array - card crop, color crop, number crop
def getCrops(frame):
    contour = getContour(frame)
    (x,y),(MA,ma), rangle = cv.fitEllipse(contour)
    ch = int(MA/1.2) #files MA = 451
    cw = int(ma/1.28) #files ma = 282
    cx = int((x - ch/2))
    cy = int((y - cw/2))
    rCardCrop = frame[cx:cx+cw, cy:cy+ch]
    clx = int(ma/4.7)
    cly = int(MA/2.5)
    clw = clh = 2
    colorCrop = rCardCrop[clx:clx+clw, cly:cly+clh]
    nx = int(ma/4.5)
    ny = int(MA/5.8)
    nw = nh = 150
    numberCrop = rCardCrop[nx:nx+nw, ny:ny+nh] 
    
    return [rCardCrop, colorCrop, numberCrop]

#rotate frame / args: numyp array - contours / return - numpy array frame
def getRotatedFrame(frame):
    cnt = getContour(frame)
    (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
    (h, w) = frame.shape[:2]
    center = (x, y)
    if angle > 90: angle = 180 + abs(angle)
    scale = 1
    M = cv.getRotationMatrix2D(center, angle, scale)
    rFrame = cv.warpAffine(frame, M, (w, h))
    return rFrame

#plot images / args: numpy arrays - frame, rotated frame, card crop, color crop, number crop
def printData(frame, rFrame, cardCrop, colorCrop, numberCrop):
    print(plt.imshow(frame))
    plt.pause(0.001)
    print(plt.imshow(rFrame))
    plt.pause(0.001)
    print(plt.imshow(cardCrop))
    plt.pause(0.001)
    print(plt.imshow(colorCrop))
    plt.pause(0.001)
    print(plt.imshow(numberCrop, cmap='Greys_r'))
    plt.pause(0.001)

#get card color / args: KNN - classifier, numpy array - card / return: string - color
def getCardColor(clf, card):
    color = ""
    prediction = clf.predict(card.data)
    color += colors[prediction[0]]
    return color

#get card number / args: KNN - classifier, numpy array - card / return: string - color
def getCardNumber(clf, card):
    number = ""
    prediction = clf.predict(card.data)
    number += str(cardNumbers[prediction[0]])
    return number

#get card data / args: Boolean - live stream, numpy array - image / return: numpy array - image, string - card info
def getCardData(isLive, frame):
    cardData = ""
    cnt = getContour(frame)
    rFrame = getRotatedFrame(frame)
    rCnt = getContour(rFrame)
    rCardCrop, colorCrop, numberCrop = getCrops(rFrame) #crops
    if numberCrop.shape == (150,150,3): 
        numberCrop = getThreshold(numberCrop)
        printData(frame, rFrame, rCardCrop, colorCrop, numberCrop) #log
        #predict
        pcol = Bunch(data=colorCrop)
        pcol = (pcol.data).reshape(1, 2*2*3)
        col = getCardColor(colorClf, pcol)
        pnum = Bunch(data=numberCrop)
        pnum = (pnum.data).reshape(1, 150*150)
        num = str(getCardNumber(numberClf, pnum))
        cardData = num + " " + col
    #draw
    x,y,w,h = cv.boundingRect(np.asarray(cnt))
    cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)
    cv.putText(frame, cardData, (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (38, 38, 38), 2)
    cv.drawContours(frame, cnt, -1, (255,0,0), 1)
    cv.drawContours(frame, rCnt, -1, (0,0,255), 1)
    return [frame, cardData]

#training models for KNN classifiers
def trainModels():
    global colorClf
    global numberClf

    #color
    print("")
    print("Loading 5200 color images...")
    dc = []
    tc = []
    for n in cardNumbers:
        t = []
        for c in cardColors:
            try:
                el = np.asarray(load('img/'+ c + str(n) +'.jpg', target_size=(resX,resY)))
                cardCrop, colorCrop, numberCrop = getCrops(getRotatedFrame(el))
                #dc.append(colorCrop)
                #t.append(cardColors.index(c))
                for i in range(0, 100):
                    dc.append(getColorAugm(colorCrop))
                    t.append(cardColors.index(c))
            except IOError: break
        tc += t
    colD = Bunch(data=np.array(dc), target=np.array(tc))
    X = colD.data.reshape(len(dc), 2*2*3)
    y = colD.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, train_size=.75, random_state=42)
    print(np.shape(X_train), np.shape(X_test), len(y_train), len(y_test))
    colorClf.fit(X_train, y_train)
    colorTrainAcc = colorClf.score(X_train, y_train)
    colorTestAcc = colorClf.score(X_test, y_test)
        
    #numbers
    print("")
    print("Loading 5200 real + augmented numbers images...")
    dn = []
    tn = []
    for c in cardColors:
        t = []
        for n in cardNumbers:
            try:
                el = np.asarray(load('img/'+ c + str(n) +'.jpg', target_size=(resX,resY)))
                cardCrop, colorCrop, numberCrop = getCrops(getRotatedFrame(el))
                numberCrop = getThreshold(numberCrop)
                cardCrop = el[225:590, 225:470]
                numberCrop = getThreshold(cardCrop[110:260, 50:200])
                #dn.append(numberCrop)
                #t.append(cardNumbers.index(n))
                dn += getNoiseAugm(numberCrop, 100)
                for i in range(0, 100):
                    t.append(cardNumbers.index(n))
            except IOError: break
        tn += t
    
    #create data sets for original and each agumentation
    numD = Bunch(data=np.array(dn), target=np.array(tn)) 
    X = numD.data.reshape(len(dn), 150*150)
    y = numD.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, train_size=.75, random_state=42)
    print(np.shape(X_train), np.shape(X_test), len(y_train), len(y_test))
    print("")
    print("Displaying real and augmented samples...")
    numberClf.fit(X_train, y_train)
    numberTrainAcc = numberClf.score(X_train, y_train)
    numberTestAcc = numberClf.score(X_test, y_test)
    
    #log ml data
    i = 0
    for i in range(0, 400):
        if (i > -1 and i < 10) or (i > 100 and i < 110) or (i > 200 and i < 210) or (i > 300 and i < 310):
            print(plt.imshow(dc[i]))
            plt.pause(0.001)
        i+=1
    i = 0
    for i in range(0, 1600):
        if (i > -1 and i < 10) or (i > 100 and i < 110) or (i > 200 and i < 210) or (i > 300 and i < 310):
            print(plt.imshow(dn[i], cmap='Greys_r'))
            plt.pause(0.001)  
        i+=1
    
    #log training, testing results
    print("")
    print("Train and Test color and number models...")
    print("100%")
    sleep(2)
    print("")
    print("ML results...")
    print("COLORS    train-acc " + ('{:.2f}'.format(colorTrainAcc)) + "       test-acc " + ('{:.2f}'.format(colorTestAcc)) )
    print("NUMBERS   train-acc " + ('{:.2f}'.format(numberTrainAcc)) + "       test-acc " + ('{:.2f}'.format(numberTestAcc)) )
   
  

    