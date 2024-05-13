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
vidCap.set(3, 1280)
vidCap.set(4, 720)

detector = htm.HandTrackingModule(detectionConf=0.85)

while True:
    # Import Image
    success, img = vidCap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        topOfIndexFingerX, topOfIndexFingerY = lmList[8][1], lmList[8][2]
        topOfMiddleFingerX, topOfMiddleFingerY = lmList[12][1], lmList[12][2]

        # Check which fingers are up

        fingers = detector.fingersUp()
        print(fingers)

        # If Selection Mode - Two fingers up

    # If Drawing Mode - Index Finger up

    # Setting the header image
    img[0:100, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
