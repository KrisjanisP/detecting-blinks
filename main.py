
# Importing the required dependencies
import cv2  # for video rendering
import dlib  # for face and landmark detection
import imutils
from imutils import face_utils 
import math

cam = cv2.VideoCapture(0)

# defining a function to calculate the EAR

def distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

def calculate_EAR(eye):
    y1 = distance(eye[1], eye[5])
    y2 = distance(eye[2], eye[4])
    x1 = distance(eye[0], eye[3])
    return (y1+y2) / x1


# Variables
blink_thresh = 0.45
succ_frame = 2
count_frame = 0

# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initializing the Models for Landmark and
# face Detection
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor(
    'model.dat')

blinks = 0

while 1:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=640)

    # converting frame to gray scale to
    # pass to detector
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting the faces
    faces = detector(img_gray)
    for face in faces:

        cv2.rectangle(frame,(face.left(),face.top()), (face.right(),face.bottom()),(0,255,0))
        
        shape = landmark_predict(img_gray, face)

        shape = face_utils.shape_to_np(shape)

        lefteye = shape[L_start:L_end]
        righteye = shape[R_start:R_end]

        cv2.polylines(frame,[lefteye],True,(255,0,0))
        cv2.polylines(frame,[righteye],True,(255,0,0))

        # Calculate the EAR
        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)

        # Avg of left and right eye EAR
        avg = (left_EAR+right_EAR)/2
        
        if avg < blink_thresh:
            count_frame += 1  # incrementing the frame count
        else:
            if count_frame >= succ_frame:
                blinks += 1
                print("blinked")
                cv2.putText(frame, str(blinks), (30, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                count_frame = 0
            else:
                count_frame = 0

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
