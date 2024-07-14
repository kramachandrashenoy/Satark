import cv2
import numpy as np
import dlib
from keras.models import load_model
from pygame import mixer
import os
import time
from imutils import face_utils

# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Pygame mixer for alarm sound
mixer.init()

# Load your custom ResNet50 model
model_resnet50 = load_model(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\weights_resnet50.h5")

# Define class labels based on your model's output
class_labels = [
    "normal driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger"
]

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\models1\shape_predictor_68_face_landmarks.dat")

def mild_alarm():
    mixer.music.load(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\alarm.mp3")
    mixer.music.play()

def harsh_alarm():
    mixer.music.load(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\alarm.mp3")
    mixer.music.play()

def no_eyes_alarm():
    mixer.music.load(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\alarm.mp3")
    mixer.music.play()

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    mar = (A + B + C) / 3.0
    return mar

def detect_drowsiness(frame, eyes_closed_start_time, eyes_detected_duration_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    ear_threshold = 0.25  # Adjust this value as needed
    mar_threshold = 0.7   # Adjust this value as needed
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

        if ear < ear_threshold:
            eyes_detected = True
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            elif time.time() - eyes_closed_start_time >= eyes_detected_duration_threshold:
                harsh_alarm()  # Use harsh alarm for prolonged eyes closed
        else:
            eyes_closed_start_time = None
        
        if mar > mar_threshold:
            harsh_alarm()  # Use harsh alarm for mouth open (yawning)

    if not eyes_detected:
        if eyes_closed_start_time is None:
            eyes_closed_start_time = time.time()
        elif time.time() - eyes_closed_start_time >= eyes_detected_duration_threshold:
            no_eyes_alarm()  # Use no eyes detected alarm if eyes not detected

    return frame, eyes_closed_start_time

def recognize():
    cap = cv2.VideoCapture(0)
    eyes_closed_start_time = None
    eyes_detected_duration_threshold = 5  # seconds
    last_capture_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        current_time = time.time()
        if current_time - last_capture_time >= 5:  # Capture one image per 5 seconds
            last_capture_time = current_time

            # Preprocess the frame and predict using your custom ResNet50 model
            # Assuming your ResNet50 model expects input of shape (224, 224,3)
            resized_frame = cv2.resize(frame, (224, 224))
            x = np.expand_dims(resized_frame, axis=0)
            x = x / 255.0  # Normalize pixel values if necessary

            # Predict the behavior
            predicted_class = np.argmax(model_resnet50.predict(x), axis=-1)
            predicted_class = int(predicted_class)  # Ensure it's converted to an integer
            
            # Example: Display predicted class label on the frame
            cv2.putText(frame, f"Behavior: {class_labels[predicted_class]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Detect drowsiness and handle alarms
            frame, eyes_closed_start_time = detect_drowsiness(frame, eyes_closed_start_time, eyes_detected_duration_threshold)

            cv2.imshow('Behavior Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
