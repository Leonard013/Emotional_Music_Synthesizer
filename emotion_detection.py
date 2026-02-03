import cv2
import dlib
import numpy as np

from config import SMILE_THRESHOLD, LANDMARK_PREDICTOR_PATH


def is_smiling(landmarks) -> bool:
    """Classify whether a face is smiling based on mouth landmark aspect ratio.

    Uses dlib 68-point landmarks: points 48, 54 (mouth corners),
    51, 57 (mouth top/bottom).

    Returns True if aspect_ratio < SMILE_THRESHOLD (default 0.2).
    """
    left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
    right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
    top_mouth = (landmarks.part(51).x, landmarks.part(51).y)
    bottom_mouth = (landmarks.part(57).x, landmarks.part(57).y)

    horizontal_dist = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
    vertical_dist = np.linalg.norm(np.array(top_mouth) - np.array(bottom_mouth))

    aspect_ratio = vertical_dist / horizontal_dist
    return aspect_ratio < SMILE_THRESHOLD


def detect_emotion_from_image(image_path: str) -> int:
    """Detect emotion from a face photo.

    Args:
        image_path: path to a photo file (e.g. photo.jpg).

    Returns:
        +1 for happy (smiling), -1 for sad (not smiling).

    Raises:
        ValueError: if no face is detected in the image.
    """
    photo = cv2.imread(image_path)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)

    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("No face detected in image. Please try again.")

    face = faces[0]
    landmarks = predictor(gray, face)

    label = 1 if is_smiling(landmarks) else -1
    return label


def detect_emotion_from_webcam() -> int:
    """Capture a photo from the webcam and detect emotion.

    Opens a window showing the camera feed. Press SPACE to capture,
    ESC to cancel.

    Returns:
        +1 for happy (smiling), -1 for sad (not smiling).

    Raises:
        ValueError: if no face is detected or capture is cancelled.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Cannot open webcam.")

    print("Press SPACE to capture, ESC to cancel.")
    captured = False
    frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture - SPACE to take photo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            captured = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if not captured or frame is None:
        raise ValueError("Photo capture cancelled.")

    cv2.imwrite("photo.jpg", frame)
    return detect_emotion_from_image("photo.jpg")
