import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from utils.cut_eyes import cut_eye_from_file
import matplotlib.pyplot as plt

def create_model(TARGET_SIZE, DIRECTION_NAMES, test = False):
    model_path = "eyeDirectionClassification.model"

    #checking if model already exists
    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print("Model loaded successfully.")
        print("======\n\n\n\n")
    else:
        images = []
        image_labels = []

        label_encoder = LabelEncoder()
        path = "./eyeTrainDataset"
    
        # taking all the images in directory, applaying color filter, normalizing, resizing and adding to the list
        for p in Path(path).glob("*.png"):
            img = cut_eye_from_file(str(p), False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(img, TARGET_SIZE)
            resized_img = resized_img / 255.0
            normalized_train_img = (resized_img - np.mean(resized_img)) / np.std(resized_img)
            images.append(normalized_train_img)
            image_labels.append(str(p.stem).split(" ")[0])

        images = np.array(images)
        images = np.expand_dims(images, axis=-1)
        image_labels = label_encoder.fit_transform(image_labels)

        # creating model
        model = models.Sequential()

        model.add(layers.Conv2D(32, (10, 10), activation='relu', input_shape=(240, 240, 1)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (10, 10), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128, (10, 10), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(256, (10, 10), activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # training model
        model.fit(images, image_labels, epochs=20)

        # saving model to the folder
        model.save(model_path)

    # testing model on the image. Image to test can be selected from any image in directory or new one can be created 
    if test:
        test_image = cut_eye_from_file("./eyeTrainDataset/left 2.png", False)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image = cv2.resize(test_image, TARGET_SIZE)
        test_image = test_image / 255.0
        normalized_test_img = (test_image - np.mean(test_image)) / np.std(test_image)
        normalized_test_img = np.expand_dims(normalized_test_img, axis=0)

        test = model.predict(normalized_test_img)

        index = np.argmax(test)

        print(f"Prediction: {DIRECTION_NAMES[index]}")

    return model