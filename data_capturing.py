import cv2
from time import sleep


def collectData(h, w):
    cardColors = ["r", "g", "b", "y"]
    cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]
    cardGeneralWilds = ["w1", "w2", "w3"]
    
    vc = cv2.VideoCapture(1)
    vc.set(3, h)
    vc.set(4, w)
    key = "key" 
    
    for c in cardColors:
        for n in cardNumbers: 
            while vc.isOpened():    
                rval, frame = vc.read()
                cv2.imshow("stream", frame)
                print('Please show: '+c+str(n))
                #take screnshot and save to file
                while key != "":
                    sleep(1/100)
                    key = input()
                key = "key"  
                cv2.imwrite('./images/'+ c + str(n)+'.jpg', frame)
                print('Saved: '+c+str(n))
    cv2.destroyWindow("stream")
    cv2.VideoCapture(0).release()








