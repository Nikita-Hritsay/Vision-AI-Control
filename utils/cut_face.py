import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

## function that cuts image by eye countor from image file
def cut_eye_from_file(path, showResult = True):
    frame = cv2.imread(str(path), cv2.CAP_DSHOW)
    return cut_eye( frame, showResult)


## for better separating - gray color should be used in the picture (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
def cut_eye(frame, showResult = True):
    
    frame = cv2.convertScaleAbs(frame)
    eyes = eye_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    if len(eyes) > 0:
        (x, y, w, h) = eyes[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame = frame[y+10:y+h-10, x+10:x+w-10]
        if showResult:
            cv2.imshow('Eye Tracking', frame)
            cv2.waitKey(0)

    return frame