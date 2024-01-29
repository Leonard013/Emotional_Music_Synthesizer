import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file


def is_smiling(landmarks):
    # Get the coordinates of the mouth landmarks
   # Points for the left and right of the mouth
    left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
    right_mouth = (landmarks.part(54).x, landmarks.part(54).y)

    # Points for the top and bottom of the mouth
    top_mouth = (landmarks.part(51).x, landmarks.part(51).y)
    bottom_mouth = (landmarks.part(57).x, landmarks.part(57).y)

    # Calculate distances
    horizontal_dist = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
    vertical_dist = np.linalg.norm(np.array(top_mouth) - np.array(bottom_mouth))

    # Aspect ratio
    aspect_ratio = vertical_dist / horizontal_dist

    # Threshold can be adjusted based on empirical data
    smile_threshold = 0.30

    return aspect_ratio < smile_threshold



# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)


        # Assuming the landmarks are numpy arrays
        if is_smiling(landmarks):

            color = (0, 255, 0) # Green
        else:
            color = (0, 0, 255) # Red
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display the resulting frame
    cv2.imshow('Smile Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
