import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.cut_eyes import cut_eye
from utils.eyeDiractionClassificationModel import create_model

TARGET_SIZE = (240, 240)
DIRECTION_NAMES = ["down", "left", "right", "up"]

if __name__ == "__main__":    
    model = create_model(TARGET_SIZE, DIRECTION_NAMES, False)
    prediction = 0
    frame_count = 0
    i = 0

    # change to 0 if only one camera is available 
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        #cv2.imshow('Eye Tracking', frame)
        frame_count += 1

        if frame_count == 10:
            frame_count = 0
            i += 1

            eye = cut_eye(frame, False)

            if len(eye) > 0:
                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                gray_eye = cv2.resize(gray_eye, TARGET_SIZE)
                gray_eye = gray_eye / 255.0
                normalized_img = (gray_eye - np.mean(gray_eye)) / np.std(gray_eye)
                normalized_img = np.expand_dims(normalized_img, axis=0)
                prediction = model.predict(normalized_img)

                prediction_index = np.argmax(prediction)

                plt.subplot(4, 4, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.squeeze(normalized_img), cmap=plt.cm.binary)
                plt.xlabel(DIRECTION_NAMES[prediction_index])

                print(f"Prediction: {DIRECTION_NAMES[prediction_index]}  {prediction}")

        if i == 10:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    plt.show()

    cap.release()
    cv2.destroyAllWindows()
