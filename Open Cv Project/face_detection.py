import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(r"C:\Users\Vishwak\Desktop\opencv\Images\video\videoplayback (1).mp4")
mpDraw = mp.solutions.drawing_utils
mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection(model_selection=1)

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face.process(imgRGB)

    if result.detections:
        for id,  detection in enumerate(result.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.location_data)
            # print(detection.location_data.relative_bounding_box)
            # print(detection.score)

            bboxC = detection.location_data.relative_bounding_box

            ih, iw,ic = img.shape # numpy ka function use ho ra to usme pehle height phr width
            bbox = int(bboxC.xmin * iw) ,int(bboxC.ymin*ih), \
                   int(bboxC.width * iw) , int(bboxC.height*ih)

            cv2.rectangle(img,bbox ,color = (255,0,255), thickness=2)
            cv2.putText(img, str(int(detection.score[0] * 100))+"%", (bbox[0], bbox[1]-28),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (35, 35, 155), 1)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (35, 35, 155), 3)

    cv2.imshow("Video", img)
    cv2.waitKey(1)