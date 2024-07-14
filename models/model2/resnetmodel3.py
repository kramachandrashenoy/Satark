import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer
import os
from keras.preprocessing.image import ImageDataGenerator

# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Pygame mixer for alarm sound
mixer.init()

# Load your custom model
model = load_model(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\weights_resnet50.h5")

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

# Data augmentation for improving model robustness
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def preprocess_frame(frame):
    # Resize frame to match model input size
    resized_frame = cv2.resize(frame, (224, 224))
    # Convert frame to numpy array and normalize pixel values
    x = np.array(resized_frame, dtype=np.float32)
    x = x / 255.0  # Normalization step if required by the model
    x = np.expand_dims(x, axis=0)  # Expand dimensions to match model input shape
    return x

def recognize():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Apply data augmentation
        augmented_frame = next(datagen.flow(preprocessed_frame, batch_size=1))[0]

        # Predict the behavior
        predictions = model.predict(np.expand_dims(augmented_frame, axis=0))
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Apply a confidence threshold
        confidence_threshold = 0.6
        if np.max(predictions) < confidence_threshold:
            predicted_class_label = "Uncertain"
        else:
            predicted_class_label = class_labels[predicted_class]

        # Display raw prediction probabilities (for debugging)
        print(f"Prediction probabilities: {predictions}")

        # Display predicted class label on the frame
        cv2.putText(frame, f"Behavior: {predicted_class_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Behavior Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
