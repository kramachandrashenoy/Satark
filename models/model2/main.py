# Continuous Video

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread

def start_alarm(sound_path):
    """Play the alarm sound"""
    playsound(sound_path)

# Load models and cascades
classes = ['Closed', 'Open']
face_cascade = cv2.CascadeClassifier(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\haarcascade_righteye_2splits.xml")
model = load_model(r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\drowiness_new7.h5")
alarm_sound = r"C:\Users\Ramachandra\OneDrive\Desktop\driver Model 1\alarm.mp3"

# Initialize variables
cap = cv2.VideoCapture(0)
count = 0
alarm_on = False
status1 = ''
status2 = ''

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in left_eye:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                eye = roi_color[ey:ey + eh, ex:ex + ew]
                eye = cv2.resize(eye, (145, 145))
                eye = eye.astype('float') / 255.0
                eye = img_to_array(eye)
                eye = np.expand_dims(eye, axis=0)
                pred = model.predict(eye)
                status1 = np.argmax(pred)
                break

            for (ex, ey, ew, eh) in right_eye:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                eye = roi_color[ey:ey + eh, ex:ex + ew]
                eye = cv2.resize(eye, (145, 145))
                eye = eye.astype('float') / 255.0
                eye = img_to_array(eye)
                eye = np.expand_dims(eye, axis=0)
                pred = model.predict(eye)
                status2 = np.argmax(pred)
                break

            if status1 == 0 and status2 == 0:  # Both eyes closed
                count += 1
                cv2.putText(frame, f"Eyes Closed, Frame count: {count}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                if count >= 10:
                    cv2.putText(frame, "Drowsiness Alert!!!", (100, height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    if not alarm_on:
                        alarm_on = True
                        t = Thread(target=start_alarm, args=(alarm_sound,))
                        t.daemon = True
                        t.start()
            else:
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                count = 0
                alarm_on = False

        cv2.imshow("Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()
