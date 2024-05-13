import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "HeaderImages"
myList = os.listdir(folderPath)

overlayList = []
for imgPath in myList:
    image = cv2.imread(f"{folderPath}/{imgPath}")
    overlayList.append(image)

header = overlayList[0]

vidCap = cv2.VideoCapture(0)
vidCap.set(3, 1920)
vidCap.set(4, 720)

while True:
    success, img = vidCap.read()

    # Setting the header image
    img[0:200, 0:1920] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
