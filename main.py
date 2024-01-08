import cv2
import os
from pathlib import Path
import numpy as np
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

def create_model():
    images = []
    image_labels = []

    label_encoder = LabelEncoder()
    path = "./eyeTrainDataset"

    for p in Path(path).glob("*.png"):

        ## Todo cut face to train only on face parts and notfull images
        images.append(cv2.imread(str(p)))
        image_labels.append(str(p.stem))

    images = np.array(images)
    image_labels = label_encoder.fit_transform(image_labels)

    model_path = "gazeBasicClassification.model"

    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print("Model loaded successfully.")
    else:

        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(720, 1280, 3)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(images, image_labels, epochs=4)

        model.save("gazeBasicClassification.model")

    test_image = cv2.resize(cv2.imread("./eyeTrainDataset/down.png"), (1280, 720))
    test_image = np.expand_dims(test_image, axis=0)  # Add an extra dimension

    test = model.predict(test_image)

    index = np.argmax(test)

    print(f"Prediction {image_labels[index]}")
    

    return model

create_model()

if __name__ == "__main_":
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)



    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.circle(roi_color, (ex + ew // 2, ey + eh // 2), 10, (0, 255, 0), 1)

        cv2.imshow('Eye Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
