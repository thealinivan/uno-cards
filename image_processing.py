#image processing file
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from keras.preprocessing.image import load_img as load 
from sklearn.utils import Bunch

cardColors = ["r", "y", "g", "b"]
cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]
colorClf = KNeighborsClassifier(2)
numberClf = KNeighborsClassifier(2)

#get card color / args: KNN - classifier, numpy array - card / return: string - color
def getCardColor(clf, card):
    color = ""
    prediction = clf.predict(card.data)
    color += cardColors[prediction[0]]
    return color

#get card number / args: KNN - classifier, numpy array - card / return: string - color
def getCardNumber(clf, card):
    number = ""
    prediction = clf.predict(card.data)
    number += str(cardNumbers[prediction[0]])
    return number

#get card data / args: numpy array - image / return: numpy array - image, string - card info
def getCardData(frame):
    #colorCrop = cardCrop[100:101, 100:101]
    #numberCrop = cardCrop[20:90, 23:73] #70x50 corner ML not square crop
    #numberCrop = cardCrop[110:260, 50:200] #150x150 middle ML square crop
    cardData = ""
    #img processing
    frame = np.asarray(frame)
    mono = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.blur(mono, (10, 10))
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35,2)
    kernel = np.ones((5, 5), np.uint8)
    close = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
    canny = cv.Canny(close, 100, 200)
    cont, t = cv.findContours(close, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #predict
    pc = Bunch(data=np.array(frame))
    pc = (pc.data).reshape(1, -1)
    col = getCardColor(colorClf, pc)
    num = str(getCardNumber(numberClf, pc))
    cardData = col + num
    #draw
    cnt = ()
    for c in cont: 
        if len(c) > len(cnt) and len(c) < (256*4): cnt = c
    #cv.drawContours(frame, cnt, -1, (0, 255, 0), 1)
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 153), 2)
    #cv.rectangle(frame, (x, y-2), (x + w, y - 30), (32, 66, 168), -1)
    cv.putText(frame, cardData, (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 153), 2)
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
    for c in cardColors:
        t = []
        for n in cardNumbers:
            try:
                el = np.asarray(load('img/'+ c + str(n) +'.jpg', target_size=(256,256)))
                dc.append(el)
                t.append(cardColors.index(c))
            except IOError: break
        tc += t
    colD = Bunch(data=np.array(dc), target=np.array(tc))
    X = colD.data.reshape(len(dc), 256*256*3)
    y = colD.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=42)
    print(np.shape(X_train), np.shape(X_test), len(y_train), len(y_test))
    colorClf.fit(X_train, y_train)
    trainAcc = colorClf.score(X_train, y_train)
    testAcc = colorClf.score(X_test, y_test)
    print("   train-acc " + str(trainAcc) + "       test-acc " + str(testAcc) )
        
    #numbers
    dn = []
    tn = []
    for n in cardNumbers:
        t = []
        for c in cardColors:
            try:
                el = np.asarray(load('img/'+ c + str(n) +'.jpg', target_size=(256,256)))
                dn.append(el)
                t.append(cardNumbers.index(n))
            except IOError: break
        tn += t
    numD = Bunch(data=np.array(dn), target=np.array(tn))
    X = (numD.data).reshape(len(dn), 256*256*3)
    y = numD.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=42)
    print(np.shape(X_train), np.shape(X_test), len(y_train), len(y_test))
    numberClf.fit(X_train, y_train)
    trainAcc = numberClf.score(X_train, y_train)
    testAcc = numberClf.score(X_test, y_test)
    print("   train-acc " + str(trainAcc) + "       test-acc " + str(testAcc) )
   
  

    