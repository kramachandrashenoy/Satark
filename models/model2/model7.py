'''
Smart Alerts:(high preference) Instead of just a generic buzzer, tailor the alerts
based on the severity of drowsiness.For short eye closures, use a mild sound or vibration.
For longer closures or wide open mouth, use a harsher sound or a combination of audio 
and visual alerts like flashing lights.'''


import cv2
import numpy as np
import dlib
from keras.models import load_model
from pygame import mixer
import os
from scipy.spatial import distance as dist
from imutils import face_utils
import time

# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize dlib's face detector, shape predictor, and drowsiness detection model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\models\shape_predictor_68_face_landmarks.dat")
model = load_model(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\drowiness_new7.h5")

# Initialize Pygame mixer for alarm sound
mixer.init()

def mild_alarm():
    mixer.music.load(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\mild_alarm.mp3")
    mixer.music.play()

def harsh_alarm():
    mixer.music.load(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\harsh_alarm.mp3")
    mixer.music.play()

def no_eyes_alarm():
    mixer.music.load(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\no_eyes_alarm.mp3")
    mixer.music.play()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (2.0 * D)
    return mar

def detect_drowsiness(frame, eyes_closed_start_time, eyes_detected_duration_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    ear_threshold = 0.25  # Threshold for EAR to consider the eyes closed
    mar_threshold = 0.7   # Adjusted threshold for MAR to consider yawning
    eyes_detected = False
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[42:48]
        rightEye = shape[36:42]
        mouth = shape[48:68]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mar = mouth_aspect_ratio(mouth)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if ear < ear_threshold:
            eyes_detected = True
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            elif time.time() - eyes_closed_start_time >= eyes_detected_duration_threshold:
                cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                harsh_alarm()
            else:
                cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                mild_alarm()
        else:
            eyes_closed_start_time = None
        
        if mar > mar_threshold:
            cv2.putText(frame, "Yawning Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            harsh_alarm()

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if not eyes_detected:
        if eyes_closed_start_time is None:
            eyes_closed_start_time = time.time()
        elif time.time() - eyes_closed_start_time >= eyes_detected_duration_threshold:
            cv2.putText(frame, "No Eyes Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            no_eyes_alarm()
    else:
        eyes_closed_start_time = None

    return frame, eyes_closed_start_time

def recognize():
    cap = cv2.VideoCapture(0)
    eyes_closed_start_time = None
    eyes_detected_duration_threshold = 5  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        frame, eyes_closed_start_time = detect_drowsiness(frame, eyes_closed_start_time, eyes_detected_duration_threshold)
        
        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
