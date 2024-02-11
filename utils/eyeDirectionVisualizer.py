import cv2

def visualize_eye_direction(frame, eye_direction):
    height, width, _ = frame.shape
    
    if eye_direction == "left":
        start_point = (0, 0)
        end_point = (width // 2, height)
    elif eye_direction == "right":
        start_point = (width // 2, 0)
        end_point = (width, height)
    elif eye_direction == "up":
        start_point = (0, 0)
        end_point = (width, height // 2)
    elif eye_direction == "down":
        start_point = (0, height // 2)
        end_point = (width, height)
    else:
        return frame

    annotated_frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), thickness=2)
    
    return annotated_frame