#data capturing file
import cv2 as cv
import matplotlib.pyplot as plt

cardColors = ["r", "y", "g", "b"]
cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]
main_src_folder = './img/'
src_folder = './train/1/'

#capture and save data:
def dataCapture(vc):
    for c in cardColors:
        for n in cardNumbers: 
            print('Please show: '+ c +str(n) )
            card = "" 
            while card == "": card = input()
            rval, frame = vc.read()
            cv.imwrite(src_folder + c + str(n)+'.jpg', frame)
            print('Saved: '+c+str(n))
            print(plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
            plt.pause(0.001)
            
#data collection for read from file image recognition
def collectData(camIndex):
    print("Opening camera with index " + str(camIndex) + "...")
    vc = cv.VideoCapture(camIndex)
    vc.set(3, 960)
    vc.set(4, 1280)
    while vc.isOpened():
        dataCapture(vc)
        k=cv.waitKey(1)
        if k==27: break
    cv.destroyAllWindows()  
    cv.VideoCapture(0).release()





