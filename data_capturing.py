#data capturing file
import cv2
from time import sleep

#data collection for read from file image recognition
def collectData():
    cardColors = ["r", "g", "b", "y"]
    cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]
    cardGeneralWilds = ["x", "y", "z"]
    
    vc = cv2.VideoCapture(1)
    vc.set(3, 969)
    vc.set(4, 1280)
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
                cv2.imwrite('./img/'+ c + str(n)+'.jpg', frame)
                print('Saved: '+c+str(n))
    cv2.destroyWindow("stream")
    cv2.VideoCapture(0).release()








