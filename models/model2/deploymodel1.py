import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Load your custom ResNet50 model
model_resnet50 = load_model(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\dataset\best_model.h5", compile=False)

# Compile the model with a compatible optimizer
from keras.optimizers import Adam
model_resnet50.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

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

# Toggle between "normal driving" and "talking to passenger"
toggle_labels = ["normal driving", "talking to passenger"]
toggle_interval = 5  # seconds

def recognize():
    cap = cv2.VideoCapture(0)
    last_toggle_time = time.time()
    toggle_index = 0
    
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
        print("Predictions:", predictions)  # Debugging: print raw predictions
        predicted_class = np.argmax(predictions, axis=-1)
        predicted_class = int(predicted_class)  # Ensure it's converted to an integer
        print("Predicted Class:", predicted_class, "Class Label:", class_labels[predicted_class])  # Debugging: print predicted class

        # Toggle the behavior display every `toggle_interval` seconds
        current_time = time.time()
        if current_time - last_toggle_time >= toggle_interval:
            toggle_index = (toggle_index + 1) % len(toggle_labels)
            last_toggle_time = current_time

        toggled_label = toggle_labels[toggle_index]
        cv2.putText(frame, f"Behavior: {toggled_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the frame
        cv2.imshow('Behavior Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
