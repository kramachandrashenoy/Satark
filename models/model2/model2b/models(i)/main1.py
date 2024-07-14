#One Photo

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import *
from PIL import Image, ImageTk
from playsound import playsound
from threading import Thread

# Load models and cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
model = load_model(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\drowiness_new7.h5")
alarm_sound = r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\alarm.mp3"

def start_alarm(sound):
    """Play the alarm sound"""
    playsound(sound)

def preprocess_eye(eye):
    eye = cv2.resize(eye, (145, 145))
    eye = eye.astype('float') / 255.0
    eye = img_to_array(eye)
    eye = np.expand_dims(eye, axis=0)
    return eye

def predict_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        
        if len(left_eye) == 0 or len(right_eye) == 0:
            return "Eyes not detected properly"

        for (x1, y1, w1, h1) in left_eye:
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = preprocess_eye(eye1)
            pred1 = model.predict(eye1)
            status1 = np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            eye2 = roi_color[y2:y2+h2, x2:x2+w2]
            eye2 = preprocess_eye(eye2)
            pred2 = model.predict(eye2)
            status2 = np.argmax(pred2)
            break

        if status1 == 0 and status2 == 0:  # Both eyes closed
            return "Drowsy"
        else:
            return "Alert"

def capture_and_process():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        status = predict_drowsiness(frame)
        print(f"Driver Status: {status}")

        if status == "Drowsy":
            t = Thread(target=start_alarm, args=(alarm_sound,))
            t.daemon = True
            t.start()

        # Display captured image in a Tkinter window
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img.thumbnail((400, 300))  # Resize image if necessary
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
    else:
        print("Failed to capture image")

    cap.release()

# Create a Tkinter window
root = Tk()
root.title("Driver Drowsiness Detection")
root.geometry("600x400")

# Create a button to capture image and process
btn_capture = Button(root, text="Capture Image", command=capture_and_process)
btn_capture.pack(pady=20)

# Create a panel to display the captured image
panel = Label(root)
panel.pack(padx=10, pady=10)

# Run the Tkinter main loop
root.mainloop()
