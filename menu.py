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
