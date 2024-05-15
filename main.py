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
    print(lmList[8][1])

    if len(lmList) != 0:
        topOfIndexFingerX, topOfIndexFingerY = lmList[8][1], lmList[8][2]
        topOfMiddleFingerX, topOfMiddleFingerY = lmList[12][1], lmList[12][2]

        # Check which fingers are up

        fingers = detector.fingersUp()

        # If Selection Mode - Two fingers up
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (topOfIndexFingerX, topOfIndexFingerY - 25),
                          (topOfMiddleFingerX, topOfMiddleFingerY + 25),
                          (255, 0, 255), cv2.FILLED)
            #print("Selection Mode")

            # Checking for click
            if topOfIndexFingerY < 100:
                if 250 < topOfIndexFingerX < 450:
                    header = overlayList[1]
                elif 550 < topOfIndexFingerX < 750:
                    header = overlayList[2]

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (topOfIndexFingerX, topOfIndexFingerY),
                       15, (255, 0, 255), cv2.FILLED)
            #print("Drawing Mode")

    # If Drawing Mode - Index Finger up

    # Setting the header image
    img[0:100, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
