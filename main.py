import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "HeaderImages"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imgPath in myList:
    image = cv2.imread(f"{folderPath}/{imgPath}")
    overlayList.append(image)

print(len(overlayList))