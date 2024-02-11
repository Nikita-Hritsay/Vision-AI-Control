import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.cut_eyes import cut_eye
from utils.eyeDiractionClassificationModel import EyeDirectionClassificationModel
from utils.eyeDirectionVisualizer import visualize_eye_direction\
    
# Target size of the cut eye image that is used to predict in the model
TARGET_SIZE = (240, 240)
# Variations of 4 main directions that can be predicted
DIRECTION_NAMES = ["down", "left", "right", "up"]
# Change to 0 if only one camera is available 
VIDEO_CAPTURE_PORT = 1
# Radius of the circle to indicate eye direction
CIRCLE_RADIUS = 20
# Circle color (BGR)
CIRCLE_COLOR = (0, 255, 0)  # Green

def run_eye_tracking():
    # Creating model 
    model = EyeDirectionClassificationModel(TARGET_SIZE, DIRECTION_NAMES, False)
    frame_count = 0
    frame_index = 0

    cap = cv2.VideoCapture(VIDEO_CAPTURE_PORT, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if frame_count == 10:
            frame_count = 0
            frame_index += 1

            # Cutting eye from the frame
            eye = cut_eye(frame, False)

            if len(eye) > 0:
                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                gray_eye = cv2.resize(gray_eye, TARGET_SIZE)
                gray_eye = gray_eye / 255.0
                normalized_img = (gray_eye - np.mean(gray_eye)) / np.std(gray_eye)
                normalized_img = np.expand_dims(normalized_img, axis=0)
                prediction = model.predict(normalized_img)
                smoothed_predictions = model.smooth_predictions(prediction)

                prediction_index = np.argmax(smoothed_predictions)
                
                annotated_frame = visualize_eye_direction(frame, prediction_index)
                
                # Calculate position for the circle based on eye direction
                if DIRECTION_NAMES[prediction_index] == "left":
                    circle_position = (50, 50)  # Example position for left direction
                elif DIRECTION_NAMES[prediction_index] == "right":
                    circle_position = (150, 50)  # Example position for right direction
                elif DIRECTION_NAMES[prediction_index] == "up":
                    circle_position = (100, 0)  # Example position for up direction
                elif DIRECTION_NAMES[prediction_index] == "down":
                    circle_position = (100, 100)  # Example position for down direction

                # Draw the circle on the frame
                cv2.circle(annotated_frame, circle_position, CIRCLE_RADIUS, CIRCLE_COLOR, -1)

                # Display the annotated frame with circle
                cv2.imshow('Eye Tracking with Circle', annotated_frame)

                # craeting subplot to show the results of predictions on the plot
                plt.subplot(4, 4, frame_index+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.squeeze(eye), cmap=plt.cm.binary)
                plt.xlabel(DIRECTION_NAMES[prediction_index])

                print(f"Prediction: {DIRECTION_NAMES[prediction_index]}  {smoothed_predictions}")

        if frame_index == 10:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":    
    run_eye_tracking()
