import cv2
import dlib
import imutils
from scipy.spatial import distance as dist 
from imutils import face_utils


def calculate_EAR(eye: list) -> float:
    if len(eye) < 6: 
        return 0
    y1 = dist.euclidean(eye[1], eye[5]) # Distance between vertical eye landmarks
    y2 = dist.euclidean(eye[2], eye[4]) 
    x1 = dist.euclidean(eye[0], eye[3]) # Distance between horizontal eye landmarks 
    EAR = (y1 + y2) / (2.0 * x1) 
    return EAR

# Initialize dlib's face detector and facial landmark predictor
detector: dlib.fhog_object_detector = dlib.get_frontal_face_detector()
predictor: dlib.shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# EAR threshold below which a blink is detected
blink_thresh: float = 0.25  

# Minimum number of consecutive frames for a blink to be registered
succ_frame: int = 2 

# Counter for consecutive frames where EAR is below the threshold
count_frame: int = 0

(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cam: cv2.VideoCapture = cv2.VideoCapture(0)

while True:
    ret: bool
    frame: cv2.Mat
    ret, frame = cam.read()
    if not ret:
        print("Failed to get frame. Exit.")
        break

    # Convert the frame to grayscale
    gray: cv2.Mat = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces: dlib.rectangles = detector(gray)

    for face in faces:
        # Predict facial landmarks for the detected face
        shape: dlib.full_object_detection = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract landmarks for the left and right eyes
        left_eye = shape[L_start:L_end]
        right_eye = shape[R_start:R_end]

        if len(left_eye) > 0 and len(right_eye) > 0:
            left_EAR: float = calculate_EAR(left_eye)
            right_EAR: float = calculate_EAR(right_eye)
            avg_EAR: float = (left_EAR + right_EAR) / 2.0

            # Check if EAR is below the blink threshold
            if avg_EAR < blink_thresh:
                count_frame += 1
            else:
                if count_frame >= succ_frame:
                    cv2.putText(frame, "Blink Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                count_frame = 0

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


