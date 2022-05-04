#main file
import sys
import matplotlib.pyplot as plt
from menu import *
from core import readFromFile
from core import liveStream
from data_capturing import collectData


cardColors = ["r", "g", "b", "y"]
cardNumbers = [0,1,2,3,4,5,6,7,8,9,"d","a","n"]

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
    











