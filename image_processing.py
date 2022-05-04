#image processing file
import matplotlib.pyplot as plt

#plot data
def plotData(frame, cardCrop, colorCrop, numberCrop):
    print(plt.imshow(frame))
    plt.pause(0.001)
    print(plt.imshow(cardCrop))
    plt.pause(0.001) 
    print(plt.imshow(colorCrop))
    plt.pause(0.001)
    print(plt.imshow(numberCrop))
    plt.pause(0.001) 
    print(colorCrop)

#get card color
def getCardColor(colorCrop, cardColors):
    color = "RED"
    return color

#get card number
def getCardNumber(numberCrop, cardNumbers):
    number = "0000"
    return number

#draw borders and respresent data on the image
def getEnhancedframe(frame, cardData):
    return frame

#get card data: color, number, special cards
def getCardData(isFromFile, frame, cardColors, cardNumbers):
    cardCrop = frame[225:590, 225:470]
    colorCrop = cardCrop[100:101, 100:101]
    numberCrop = cardCrop[110:260, 70: 180]
    if isFromFile: plotData(frame, cardCrop, colorCrop, numberCrop)
    cardData = ""
    cardData += getCardColor(colorCrop, cardColors)
    cardData += str(getCardNumber(numberCrop, cardNumbers))
    enhancedframe = getEnhancedframe(frame, cardData)
    return [enhancedframe, cardData]


    