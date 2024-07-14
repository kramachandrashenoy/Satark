# Works both when the eyes are small and when eyes are not detected

import cv2
import numpy as np
import dlib
from keras.models import load_model
from pygame import mixer
import os
from scipy.spatial import distance as dist
from imutils import face_utils

# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize dlib's face detector, shape predictor, and drowsiness detection model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\models\shape_predictor_68_face_landmarks.dat")
model = load_model(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\drowiness_new7.h5")

# Initialize Pygame mixer for alarm sound
mixer.init()

def alarm_drowsy():
    mixer.music.load(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\alarm.mp3")
    mixer.music.play()

def alarm_no_eyes():
    mixer.music.load(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\buzzer-18-203421.mp3")
    mixer.music.play()

def eye_aspect_ratio(eye):
    # Compute the distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def preprocess_face(img, face):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_img = img[y:y+h, x:x+w]
    
    if face_img.size == 0:
        return None
    
    face_img = cv2.resize(face_img, (145, 145))  # Adjust based on your model's input requirements
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    ear_threshold = 0.25  # Threshold for EAR to consider the eyes closed
    eyes_detected = False
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[42:48]
        rightEye = shape[36:42]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # If the EAR is below the threshold, the person is drowsy
        if ear < ear_threshold:
            cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            alarm_drowsy()
        else:
            cv2.putText(frame, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Eyes were detected
        eyes_detected = True
        
        # Draw a rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Display EAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if not eyes_detected:
        alarm_no_eyes()
        cv2.putText(frame, "No Eyes Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame

def recognize():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        frame = detect_drowsiness(frame)
        
        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
