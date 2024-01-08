import cv2
import numpy as numpy

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

## function that cuts image by face countor
def cut_face(path, showResult = True):

    frame = cv2.imread(str(path), cv2.CAP_DSHOW)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    (x, y, w, h) = eyes[0]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    frame = frame[y:y+h, x:x+w]

    if showResult:
        cv2.imshow('Eye Tracking', frame)
        cv2.waitKey(0)

    return frame
