#main file
from cam import openCam
from core import readFromFile
from core import liveStream

#user option
def getUserOption():
    print("")
    print ("-----------------")
    print ("*** Uno Cards ***")
    print ("-----------------")
    print("[1] Read from file")
    print("[2] Live camera stream")
    print("[3] Data capturing")
    option = input("::")
    return option

#main loop
opt = getUserOption()
while opt != "0":
    #run IR by reading from file
    if opt == "1": readFromFile()
    
    #run IR by reading opening live camera stream
    elif opt == "2": liveStream()
    
    #data capturing
    elif opt == "3": collectData()
    
    #exit
    elif opt == "0": sys.exit("Terminated") 
    else: print("Invalid option !")
    plt.pause(0.0001)
    opt = getUserOption()
    











