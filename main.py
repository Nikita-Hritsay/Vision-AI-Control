import cv2
import os
from pathlib import Path
import numpy as np
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from utils.cut_eyes import cut_eye_from_file, cut_eye

TARGET_SIZE = (240, 240)
DIRECTION_NAMES = ["down", "left", "right", "up"]


def create_model(test = False):
    model_path = "gazeBasicClassification.model"

    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print("Model loaded successfully.")
        print("======\n\n\n\n")
    else:
        images = []
        image_labels = []

        label_encoder = LabelEncoder()
        path = "./eyeTrainDataset"
    
        for p in Path(path).glob("*.png"):
            img = cut_eye_from_file(str(p), False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(img, TARGET_SIZE)
            resized_img = resized_img / 255.0
            normalized_img = (resized_img - np.mean(resized_img)) / np.std(resized_img)
            images.append(normalized_img)
            image_labels.append(str(p.stem).split(" ")[0])

        images = np.array(images)
        images = np.expand_dims(images, axis=-1)
        image_labels = label_encoder.fit_transform(image_labels)

        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(240, 240, 1)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(images, image_labels, epochs=4)

        model.save("gazeBasicClassification.model")

    if test:
        test_image = cut_eye_from_file("./eyeTrainDataset/left 2.png", False)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image = cv2.resize(test_image, TARGET_SIZE)
        test_image = test_image / 255.0
        normalized_img = (resized_img - np.mean(test_image)) / np.std(test_image)
        normalized_img = np.expand_dims(normalized_img, axis=0)

        test = model.predict(normalized_img)

        index = np.argmax(test)

        print(f"Prediction: {DIRECTION_NAMES[index]}")

    return model

if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    model = create_model(False)
    prediction = 0

    while True:
        ret, frame = cap.read()
        cv2.imshow('Eye Tracking', frame)

        eye = cut_eye(frame, False)

        if len(eye) > 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, TARGET_SIZE)
            gray_frame = gray_frame / 255.0
            normalized_img = (gray_frame - np.mean(gray_frame)) / np.std(gray_frame)
            normalized_img = np.expand_dims(normalized_img, axis=0)

            prediction = model.predict(normalized_img)

            index = np.argmax(prediction)

            print(f"Predicton values: {prediction}")

            print(f"Predicton value: {index}")

            print(f"Prediction: {DIRECTION_NAMES[index]}")


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
