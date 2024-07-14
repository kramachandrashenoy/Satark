import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer
import os
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load your custom ResNet50 model
model_resnet50 = load_model(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\weights_resnet50.h5")

# Initialize Pygame mixer for alarm sound
mixer.init()

def alarm_sound(file_path):
    mixer.music.load(file_path)
    mixer.music.play()

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

def recognize():
    cap = cv2.VideoCapture(0)
    eyes_closed_start_time = None
    eyes_detected_duration_threshold = 5  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        # Preprocess the frame for your custom ResNet50 model
        resized_frame = cv2.resize(frame, (224, 224))
        x = np.expand_dims(resized_frame, axis=0)
        x = x / 255.0  # Normalize pixel values

        # Predict the behavior
        predictions = model_resnet50.predict(x)
        predicted_class = np.argmax(predictions, axis=-1)
        predicted_class = int(predicted_class)  # Ensure it's converted to an integer
        
        # Display predicted class label on the frame
        cv2.putText(frame, f"Behavior: {class_labels[predicted_class]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Check if no eyes detected for more than the threshold duration
        if predicted_class == class_labels.index("normal driving"):
            eyes_detected = True
        else:
            eyes_detected = False
        
        if not eyes_detected:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            elif time.time() - eyes_closed_start_time >= eyes_detected_duration_threshold:
                alarm_sound(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\alarm-214447 (1).mp3")
                cv2.putText(frame, "No Eyes Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            eyes_closed_start_time = None

        # Display the frame
        cv2.imshow('Behavior Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
