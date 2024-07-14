import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing import image

import os

# Define the class labels
class_labels = {
    0: 'hair and makeup',
    1: 'talking on the phone - right',
    2: 'texting-right',
    3: 'texting - left',
    4: 'talking on the phone - left',
    5: 'operating the radio',
    6: 'drinking',
    7: 'reaching behind',
    8: 'normal driving',
    9: 'talking to passenger'
}

# Load the pre-trained model
model_path = r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\weights_resnet50.h5"  # Update with your model path
model = load_model(model_path)

def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize pixel values
    pred = model.predict(x)
    pred_class = np.argmax(pred)
    pred_label = class_labels[pred_class]
    return pred_label


# Capture video from the laptop camera
def capture_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow('Video Feed', frame)
        
        # Press 'q' to quit capturing video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to capture and predict in real-time
def real_time_prediction():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the captured frame
        cv2.imshow('Video Feed', frame)
        
        # Save the frame as an image
        img_path = 'temp_img.jpg'
        cv2.imwrite(img_path, frame)
        
        # Predict the class of the captured image
        predicted_class = predict_image_class(img_path)
        print("Predicted Class:", predicted_class)
        
        # Press 'q' to quit capturing and predicting
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time prediction function
real_time_prediction()
