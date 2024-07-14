# AI assisted distracted driver detection system

Objective: To design and implement a system that uses AI algorithms to detect distracted driving behaviours in real-time.
<br>
<br>
<b>Key Steps:</b> <br>
Step 1: Data Collection <br>
Step 2: Model Training and Fine-Tuning <br>
Step 3: Real-Time Processing <br>
Step 4: Front-End Development <br>
Step 5: Backend Development <br>
Step 6: Dashboard Development <br>

<b>Takeaways</b> <br>
We were able to train a Resnet neural network model sucessfully by gathering distracted driver dataset. <br>
We used ultra-sonic sensors to detect any nearby objects and give an alert to the driver.<br>
We were sucessful in fetching data from google APIs and recommend nearby hotels for the driver.<br>
We were able to build a database to store numerical aspects of distracted drivers and analyse the data through pie-charts.<br>
We tried exploring explainable AI and assist the drivers.<br>

<b>Block diagram</b>
![Sample Image](utils/image.jpg)

Installation and Setup
Step 1: Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/satark.git
cd satark
Step 2: Set Up the Python Environment
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Step 3: Configure MongoDB
Set up a MongoDB Atlas account if you don't already have one.
Update the MongoDB connection string in the main.py file:
python
Copy code
client = MongoClient("your_mongodb_connection_string")
Step 4: Set Up the Ultrasonic Sensors (Arduino)
Connect the ultrasonic sensors to your Arduino as per the provided code.
Upload the provided Arduino code (ultrasonic_sensor.ino) to your Arduino board using the Arduino IDE.
Ensure the correct serial port is set in the main.py file:
python
Copy code
serial_port = 'COM6'  # For Windows
# serial_port = '/dev/ttyACM0'  # For Linux/macOS
Step 5: LocationIQ API Configuration
Get your API key from LocationIQ.
Update the locationiq_api_key variable in the main.py file with your API key.
Step 6: Run the Flask Application
bash
Copy code
python main.py
The application will start on http://127.0.0.1:5000/.

Step 7: Access the Application
Open a web browser and go to http://127.0.0.1:5000/ to access the main interface.

Directory Structure
arduino
Copy code
satark/
├── static/
│   └── final.html
├── templates/
│   └── index.html
├── ultrasonic_sensor.ino
├── shape_predictor_68_face_landmarks.dat
├── drowiness_new7.h5
├── alarm.mp3
├── buzzer2-6109.mp3
├── requirements.txt
└── main.py
APIs
Video Feed
Endpoint: /video_feed
Description: Provides the real-time video feed with drowsiness detection.
Parameters: driver_id (optional, default: 1)
Prediction
Endpoint: /prediction
Description: Returns the current drowsiness detection status.
Response: { 'prediction': 'Drowsiness Detected' | 'Alert' | 'No Eyes Detected' }
Ultrasonic Data
Endpoint: /ultrasonic
Description: Returns the distance measured by the ultrasonic sensors.
Response: { 'distance1': <value>, 'distance2': <value> }
Current Driving Data
Endpoint: /current_driving_data
Description: Returns the current driving statistics for the specified driver.
Parameters: id (optional, default: 1)
Total Driving Data
Endpoint: /total_driving_data
Description: Returns the all-time driving statistics for the specified driver.
Parameters: id (optional, default: 1)
Fetch Nearby Places
Endpoint: /fetch_nearby_places
Description: Returns the nearby places of interest.
Response: { 'hotel': [...], 'restaurant': [...], 'hospital': [...] }

Acknowledgements
Dlib for facial landmark detection.
Keras for the drowsiness detection model.
MongoDB for database services.
LocationIQ for location services.
Contributing
Contributions are welcome! Please open an issue or submit a pull request.
