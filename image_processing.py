#image processing file
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from keras.preprocessing.image import load_img as load 
from sklearn.utils import Bunch

cardColors = ["r", "g", "b", "y"]
colors = ["RED","GREEN", "BLUE", "YELLOW"]
cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]
colorClf = KNeighborsClassifier(2)
numberClf = KNeighborsClassifier(2)
resX = 600
resY = 800

#get contours / args: numpy array - image / return numpy array - contours
def getContours(frame):
   #contours of card in frame
   mono = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
   blur = cv.blur(mono, (10, 10))
   th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 2)
   kernel = np.ones((5, 5), np.uint8)
   close = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
   #canny = cv.Canny(close, 50, 100)
   cont, t = cv.findContours(close, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
   return cont

def getThreshold(frame):
    mono = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.blur(mono, (5, 5))
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, 2)
    return th

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

#get card data / args: numpy array - image / return: numpy array - image, string - card info
def getCardData(frame):
    cardData = ""
    
    #card in image
    cardCont = getContours(frame)
    cnt = ()
    maxCnt = 1600
    for c in cardCont: 
        if len(c) > len(cnt) and len(c) < maxCnt: cnt = c
    (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
    
    #rotate frame
    (h, w) = frame.shape[:2]
    center = (x, y)
    angle = 360 - (180 - angle)
    scale = 1
    M = cv.getRotationMatrix2D(center, angle, scale)
    rFrame = cv.warpAffine(frame, M, (w, h))
    
    #card in rotated image
    rCardCont = getContours(rFrame)
    rCnt = ()
    maxCnt = 1600
    for c in rCardCont: 
        if len(c) > len(rCnt) and len(c) < maxCnt: rCnt = c
    (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
    
    #crops
    ch = 360
    cw = 240
    cx = int((x - (ch-120)/2))
    cy = int((y - (cw+120)/2))
    cardCrop = rFrame[cx:cx+ch, cy:cy+cw]
    #numberCrop = cardCrop[15:95, 15:95] 80x80 corner
    numberCrop = cardCrop[110:260, 50:200]
    colorCrop = cardCrop[100:102, 150:152]
    
    #number in card
    numberCrop = getThreshold(numberCrop)
   
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
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 153), 2)
    cv.putText(frame, cardData, (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (38, 38, 38), 2)
    
    return [frame, cardData]

#training models for KNN classifiers
def trainModels():
    global colorClf
    global numberClf
    trainAcc = 0
    testAcc = 0
    
    #color
    dc = []
    tc = []
    for n in cardNumbers:
        t = []
        for c in cardColors:
            try:
                el = np.asarray(load('img/'+ c + str(n) +'.jpg', target_size=(resX,resY)))
                cardCrop = el[225:590, 225:470]
                colorCrop = cardCrop[100:102, 100:102]
                #print(plt.imshow(colorCrop))
                #plt.pause(0.001)
                dc.append(colorCrop)
                t.append(cardColors.index(c))
            except IOError: break
        tc += t
    colD = Bunch(data=np.array(dc), target=np.array(tc))
    X = colD.data.reshape(len(dc), 2*2*3)
    y = colD.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=42)
    print(np.shape(X_train), np.shape(X_test), len(y_train), len(y_test))
    colorClf.fit(X_train, y_train)
    colorTrainAcc = colorClf.score(X_train, y_train)
    colorTestAcc = colorClf.score(X_test, y_test)
        
    #numbers
    dn = []
    tn = []
    for n in cardNumbers:
        t = []
        for c in cardColors:
            try:
                el = np.asarray(load('img/'+ c + str(n) +'.jpg', target_size=(resX,resY)))
                cardCrop = el[225:590, 225:470]
                numberCrop = getThreshold(cardCrop[110:260, 50:200])
                #print(plt.imshow(numberCrop, cmap='Greys_r'))
                #plt.pause(0.001)
                dn.append(numberCrop)
                t.append(cardNumbers.index(n))
            except IOError: break
        tn += t
    numD = Bunch(data=np.array(dn), target=np.array(tn))
    X = numD.data.reshape(len(dc), 150*150)
    y = numD.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=42)
    print(np.shape(X_train), np.shape(X_test), len(y_train), len(y_test))
    numberClf.fit(X_train, y_train)
    numberTrainAcc = numberClf.score(X_train, y_train)
    numberTestAcc = numberClf.score(X_test, y_test)
    
    print("   train-acc " + str(colorTrainAcc) + "       test-acc " + str(colorTestAcc) )
    print("   train-acc " + str(numberTrainAcc) + "       test-acc " + ('{:.2f}'.format(numberTestAcc)) )
   
  

    