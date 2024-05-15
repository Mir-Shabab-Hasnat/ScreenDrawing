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
drawColor = (255, 0, 255)

vidCap = cv2.VideoCapture(1)
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

        # If Selection Mode - Two fingers up
        if fingers[1] and fingers[2]:

            #print("Selection Mode")

            # Checking for click
            if topOfIndexFingerY < 100:
                if 145 < topOfIndexFingerX < 215:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 460 < topOfIndexFingerX < 510:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)

                elif 780 < topOfIndexFingerX < 825:
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 1000 < topOfIndexFingerX < 1085:
                    header = overlayList[4]
                    drawColor = (255, 255, 255)

            cv2.rectangle(img, (topOfIndexFingerX, topOfIndexFingerY - 25),
                          (topOfMiddleFingerX, topOfMiddleFingerY + 25),
                          drawColor, cv2.FILLED)


        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (topOfIndexFingerX, topOfIndexFingerY),
                       15, drawColor, cv2.FILLED)
            #print("Drawing Mode")

    # If Drawing Mode - Index Finger up

    # Setting the header image
    img[0:100, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
