import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
#Imports Done, now reading the csv + accuracy
    #----System Initializing... -> Done----#

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(x , y, random_state=0, train_size = 0.75, test_size = 0.25)# Try with 10725 and 3575 (0.25 and 0.75 does same thing)
xTrainScaled = x_train/255
xTestScaled = x_test/255

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(xTrainScaled, y_train)

#Function time!
    #----System Initializing... -> Done----#

cap = cv2.VideoCapture(0)

def prediction(image):
    pil = Image.open(image)

    image = pil.convert('L') #Pixel = single value from 0 to 255
    image_resized = image.resize((22,30), Image.ANTIALIAS)

    #real_image = PIL.ImageOps.invert(image_resized) #Flipping image as camera shows mirror image

    minPixel = np.percentile(image_resized, 20) #Scalar quantity conversion

    #Scales values between 0 , 255
    real_image_scaled = np.clip(image_resized - minPixel, 0, 255) 
    maxPixel = np.max(image_resized)

    image_array = np.asarray(real_image_scaled) / maxPixel #Converts into array

    test = np.array(image_array).reshape(1,660)
    testPred = clf.predict(test)

    return testPred[0]