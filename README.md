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


## Installation and Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/satark.git
cd satark

Step 2: Set Up the Python Environment
1. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

2. Install the required Python packages:

pip install -r requirements.txt

Step 3: Configure MongoDB
Set up a MongoDB Atlas account if you don't already have one.
Update the MongoDB connection string in the main.py file:

client = MongoClient("your_mongodb_connection_string")
