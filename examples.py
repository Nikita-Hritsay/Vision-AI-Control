import cv2
import numpy as numpy
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

## function that cuts image by face countor
def cut_face():
    frame = cv2.imread("./eyeTrainDataset/left.png", cv2.CAP_DSHOW)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame = frame[y:y+h, x:x+w]

    cv2.imshow('Eye Tracking', frame)

    cv2.waitKey(0)

## function that uses tensorflow to classify the image
def image_classification_using_tensorflow():
    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

    training_images, testing_images = training_images / 255, testing_images / 255


    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Hor`se', 'Ship', 'Truck']

    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(training_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[training_labels[i][0]])
        
    plt.show()


    training_images = training_images[:20000]
    training_labels = training_labels[:20000]
    testing_images = testing_images[:4000]
    testing_labels = testing_labels[:4000]

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    ## 10 because we have 10 possible outputs
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    ## saving model so there is no need to train the model every time we run the program
    model.save("image_classification.model")

    ## to get the trained model we can use this
    ## model = models.load_model("image_classification.model")



image_classification_using_tensorflow()

