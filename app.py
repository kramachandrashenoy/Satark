# main one

from flask import Flask, jsonify, Response, send_from_directory, request, render_template
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
from pymongo import MongoClient
import requests, json, math

app = Flask(__name__, static_url_path='', static_folder='static')

# MongoDB connection
client = MongoClient("mongodb+srv://prateekrjt12:3wDliiaphQzxeOi2@prateekrjt12.wmjyqrh.mongodb.net/?retryWrites=true&w=majority&appName=prateekrjt12")
db = client.driver
collection = db.driver_data

# Replace this with the correct port
serial_port = 'COM6'  # For Windows
# serial_port = '/dev/ttyACM0'  # For Linux/macOS

ser = None  # Initialize serial connection globally
latest_frame = None  # Global variable to store the latest frame

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the machine learning model
model = load_model('drowiness_new7.h5')

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

def increment_drowsiness_count(driver_id):
    collection.update_one(
        {"id": driver_id},
        {
            "$inc": {
                "current_driving.current_drowsiness_count": 1,
                "all_time_stats.drowsiness_count": 1,
                "current_driving.current_total_count": 1,
                "all_time_stats.total_count": 1
            }
        }
    )

def increment_movement_count(driver_id):
    collection.update_one(
        {"id": driver_id},
        {
            "$inc": {
                "current_driving.current_movement_count": 1,
                "all_time_stats.movement_count": 1,
                "current_driving.current_total_count": 1,
                "all_time_stats.total_count": 1
            }
        }
    )

def increment_objects_count(driver_id):
    collection.update_one(
        {"id": driver_id},
        {
            "$inc": {
                "current_driving.current_objects_count": 1,
                "all_time_stats.objects_count": 1,
                "current_driving.current_total_count": 1,
                "all_time_stats.total_count": 1
            }
        }
    )

def detect_drowsiness(frame, eyes_closed_start_time, eyes_detected_duration_threshold, driver_id):
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
            increment_drowsiness_count(driver_id)  # Increment drowsiness count in the database
            yawning_detected = True
        else:
            cv2.putText(frame, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            increment_objects_count(1)
        
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
            increment_movement_count(driver_id)  # Increment movement count in the database
            cv2.putText(frame, "No Eyes Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        eyes_closed_start_time = None

    return frame, eyes_closed_start_time

def generate_frames(driver_id):
    global latest_frame
    cap = cv2.VideoCapture(0)  # Adjust the camera index if needed
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    eyes_closed_start_time = None
    eyes_detected_duration_threshold = 1  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, eyes_closed_start_time = detect_drowsiness(frame, eyes_closed_start_time, eyes_detected_duration_threshold, driver_id)
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
    driver_id = int(request.args.get('driver_id', 1))  # Get driver ID from query parameters, default to 1
    return Response(generate_frames(driver_id), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    distance1, distance2 = read_serial_data()
    return jsonify({'distance1': distance1, 'distance2': distance2})

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
            distance_values = data.split(',')
            if len(distance_values) == 2:
                distance1 = float(distance_values[0])
                distance2 = float(distance_values[1])
                print(f"{distance1},{distance2}")
                return distance1, distance2
            else:
                return "No data available", "No data available"
        return "No data available", "No data available"
    except Exception as e:
        print(f"Error reading serial port {serial_port}: {e}")
        return "Error", "Error"

@app.route('/current_driving_data')
def current_driving_data():
    driver_id = request.args.get('id', 1)  # Get driver ID from query parameters, default to 1
    try:
        document = collection.find_one({"id": int(driver_id)})
        if document and 'current_driving' in document:
            return jsonify(document['current_driving'])
        else:
            return jsonify({"error": "Data not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/total_driving_data')
def total_driving_data():
    driver_id = request.args.get('id', 1)  # Get driver ID from query parameters, default to 1
    try:
        document = collection.find_one({"id": int(driver_id)})
        if document and 'all_time_stats' in document:
            return jsonify(document['all_time_stats'])
        else:
            return jsonify({"error": "Data not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500        

@app.route('/')
def index():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'final.html')

def reset_driver_stats(driver_id):
    collection.update_one(
        {"id": driver_id},
        {
            "$set": {
                "current_driving.current_drowsiness_count": 0,
                "current_driving.current_movement_count": 0,
                "current_driving.current_total_count": 0,
                "current_driving.current_objects_count":0
            }
        }
    )

# Function to get location coordinates from an address using LocationIQ API
def get_location(address, api_key):
    base_url = "https://us1.locationiq.com/v1/search.php"
    params = {
        "key": api_key,
        "q": address,
        "format": "json"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data:
            location = data[0]
            return {"lat": float(location["lat"]), "lon": float(location["lon"])}
        else:
            print("No results found.")
            return None
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 401:
            print("Error: Unauthorized. Check your API key.")
        else:
            print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None

# Function to calculate distance between two sets of coordinates using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# Function to find nearby places of a specific type using LocationIQ API
def find_nearby_places(lat, lon, place_type, api_key, limit=2):
    base_url = "https://us1.locationiq.com/v1/nearby.php"
    params = {
        "key": api_key,
        "lat": lat,
        "lon": lon,
        "tag": place_type,
        "radius": 10000,  # Adjust radius as needed
        "format": "json"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        places = response.json()
        
        if places:
            # Calculate the distance from the latitude and longitude for each place
            for place in places:
                place_lat = float(place['lat'])
                place_lon = float(place['lon'])
                distance = haversine(lat, lon, place_lat, place_lon)
                place['distance'] = round(distance, 2)  # Round distance to 2 decimal places
            
            # Sort the places by distance
            places.sort(key=lambda x: x['distance'])
            
            # Select the top 'limit' nearest locations
            top_places = places[:limit]
            
            # Prepare the results
            results = []
            for place in top_places:
                result = {
                    "name": place.get('name', 'Unnamed'),
                    "distance_km": place['distance']
                }
                results.append(result)
            
            return results
        
        else:
            print(f"No {place_type}s found.")
            return []
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

@app.route('/fetch_nearby_places', methods=['GET'])
def fetch_nearby_places():
    try:
        # Replace with your actual API key
        locationiq_api_key = ''
        
        # Example address to get location coordinates
        address = 'Vidyavardhaka college of engineering, Mysore, Karnataka'
        
        # Get location coordinates
        location = get_location(address, locationiq_api_key)
        
        if location:
            lat, lon = location['lat'], location['lon']
            
            categories = ['hotel', 'restaurant', 'hospital']
            
            all_results = {}
            
            for category in categories:
                # Find nearby places
                places = find_nearby_places(lat, lon, category, locationiq_api_key, limit=2)
                
                if places:
                    all_results[category] = places
                else:
                    all_results[category] = []
            
            return jsonify(all_results)
        
        else:
            return jsonify({'error': 'Location not found.'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500    

if __name__ == '__main__':
    try:
        setup_serial()  # Initialize the serial connection
        reset_driver_stats(1)
        app.run(debug=True)
    finally:
        # Ensure to close the serial port when the Flask app exits
        if ser and ser.is_open:
            ser.close()
            print(f"Closed serial port {serial_port}")    
