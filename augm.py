#augmentation script

import random
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

#add noise / args: numpy array - image
def addNoise(img):
    VARIABILITY = 10
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

#get data generator / args: string - augmentation type / return data generation - data generation
def getDataGen():
    datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            preprocessing_function=addNoise,
        )
    return datagen

#Source: https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
def getColorAugm(img):
    value = random.uniform(0.6, 1.3)
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    return img

#get augmented data / args: string - augmentation type, list - real images data, int - number of augmentation output images per image
def getNoiseAugm(img, duplication):
    images = []
    img = img.reshape((1,) + img.shape + (1,))
    datagen = getDataGen()
    for batch in datagen.flow(img, batch_size=1):
        i = image.img_to_array(image.array_to_img(batch[0])) 
        i = i.reshape((150,150))
        images.append(i)
        if len(images) >= duplication:
            break  
    return images


