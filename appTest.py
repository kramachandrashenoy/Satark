# just for testing purpose

from flask import Flask, jsonify, Response, send_from_directory
import cv2
import serial
import numpy as np
import dlib
from keras.models import load_model
from pygame import mixer
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import os

app = Flask(__name__, static_url_path='', static_folder='static')

# Replace this with the correct port
serial_port = 'COM6'  # For Windows
# serial_port = '/dev/ttyACM0'  # For Linux/macOS

ser = None  # Initialize serial connection globally
latest_frame = None  # Global variable to store the latest frame

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the machine learning model
model = load_model('satark.h5')

# Initialize Pygame mixer for alarm sound
mixer.init()

def alarm_drowsy():
    mixer.music.load(r"alarm.mp3")
    mixer.music.play()

def alarm_no_eyes():
    mixer.music.load(r"buzzer2-6109.mp3")
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
    yawning_detected = False
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[42:48]
        rightEye = shape[36:42]
        mouth = shape[48:68]  # Mouth landmarks

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mar = mouth_aspect_ratio(mouth)

        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Check for drowsiness or yawning
        if ear < ear_threshold or mar > mar_threshold:
            cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            alarm_drowsy()
            yawning_detected = True
        else:
            cv2.putText(frame, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Eyes were detected
        eyes_detected = True
        
        # Draw a rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Display EAR and MAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Alarm for no eyes detected for more than the threshold duration
    if not eyes_detected:
        if eyes_closed_start_time is None:
            eyes_closed_start_time = time.time()
        elif time.time() - eyes_closed_start_time >= eyes_detected_duration_threshold:
            alarm_no_eyes()
            cv2.putText(frame, "No Eyes Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        eyes_closed_start_time = None

    return frame, eyes_closed_start_time

def generate_frames():
    global latest_frame
    cap = cv2.VideoCapture(0)  # Adjust the camera index if needed
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    eyes_closed_start_time = None
    eyes_detected_duration_threshold = 5  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, eyes_closed_start_time = detect_drowsiness(frame, eyes_closed_start_time, eyes_detected_duration_threshold)
        latest_frame = frame  # Update the global variable with the latest frame

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            raise IOError("Failed to encode frame into JPEG")

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def get_prediction():
    global latest_frame
    if latest_frame is not None:
        gray = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        if faces:
            return jsonify({'prediction': 'Drowsiness Detected' if "Drowsiness Detected" in latest_frame else 'Alert'})
        else:
            return jsonify({'prediction': 'No Eyes Detected'})
    return jsonify({'prediction': 'Error'})

@app.route('/ultrasonic')
def get_ultrasonic_data():
    distance = read_serial_data()
    return jsonify({'distance': distance})

def setup_serial():
    global ser
    try:
        ser = serial.Serial(serial_port, baudrate=9600, timeout=1)
        print(f"Opened serial port {serial_port} at 9600 baud")
    except serial.SerialException as e:
        print(f"Could not open serial port {serial_port}: {e}")

def read_serial_data():
    global ser
    try:
        if ser and ser.is_open:
            data = ser.readline().decode('utf-8').strip()
            return data
        return "No data available"
    except Exception as e:
        print(f"Error reading serial port {serial_port}: {e}")
        return "Error"

@app.route('/')
def index():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'index.html')

if __name__ == '__main__':
    try:
        setup_serial()  # Initialize the serial connection
        app.run(debug=True)
    finally:
        # Ensure to close the serial port when the Flask app exits
        if ser and ser.is_open:
            ser.close()
            print(f"Closed serial port {serial_port}")
