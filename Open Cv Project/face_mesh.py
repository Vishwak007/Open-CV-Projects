from __future__ import division
import cv2
import time
import mediapipe as mp


cap = cv2.VideoCapture(1) #r"C:\Users\HP\PycharmProjects\opencv\Images\video\Misinterpretations.mp4"

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
DrawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius= 1)

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for lms in results.multi_face_landmarks:
            # print(lms)

            mpDraw.draw_landmarks(img, lms, mpFaceMesh.FACEMESH_CONTOURS, DrawSpec, DrawSpec)

            for id, flms in enumerate(lms.landmark):
                # print(id, flms)


                # ih = img.shape[0]
                # iw = img.shape[1]
                ih, iw, ic = img.shape
                # print(ih)

                x, y = int(flms.x*iw), int(flms.y*ih)

                print("x, y :", x, y)

    cTime = time.time()

    if (cTime - pTime) != 0:
        fps = 1.0 / (cTime - pTime)

    pTime = cTime

    cv2.putText(img, "fps:" + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Video", img)
    cv2.waitKey(1)