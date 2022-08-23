import cv2
import mediapipe as mp
import numpy as np
import time
import math


from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


im_width = 480
im_height = 620

cap = cv2.VideoCapture(1)
print(cap)
# address = "http://192.168.1.17:8080/video"
# cap.open(address)
cap.set(3, im_width)
cap.set(4, im_height)

pTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

vol = 0
vol_new = 400
per = 0



while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_RGB)

    list = []

    if result.multi_hand_landmarks:
        for each_hand_lms in result.multi_hand_landmarks:
            for id, landmarks in enumerate(each_hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(landmarks.x * w), int(landmarks.y * h)
                list.append([id, cx, cy])
                # if id == 20:
                #     cv2.circle(img, (cx, cy), 25, (255, 44, 99), cv2.FILLED)

            mpDraw.draw_landmarks(img, each_hand_lms, mpHands.HAND_CONNECTIONS)
    # print(list)

    if len(list) != 0:
        # print(list[4], list[8])

        x1, y1 = list[4][1], list[4][2]
        x2, y2 = list[8][1], list[8][2]


        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2, y2), (255, 0, 255), 3)

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2 -y1)
        # print(length)


        # hand range 50-210
        # volume range -60 - 0

        vol = np.interp(length, [50, 185], [minVol, maxVol])
        print(int(length), vol)

        volume.SetMasterVolumeLevel(vol, None)

        if length < 52:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        vol_new = np.interp(vol, [-65, 0], [400, 150])
        per= np.interp(vol, [-65, 0], [0, 100])

        cv2.rectangle(img, (50,150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50,int(vol_new)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(per))+"%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, "fps :"+ str(int(fps)), (50,60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    img = cv2.resize(img, (400, 420))
    cv2.imshow("Live", img)
    cv2.waitKey(1)