// Define pins for Sensor 1
const int trigPin1 = 9;   // Trigger pin for Sensor 1
const int echoPin1 = 10;  // Echo pin for Sensor 1

// Define pins for Sensor 2
const int trigPin2 = 11;  // Trigger pin for Sensor 2
const int echoPin2 = 12;  // Echo pin for Sensor 2

long duration1, duration2;
float distance1, distance2;

void setup() {
  Serial.begin(9600);
  pinMode(trigPin1, OUTPUT);
  pinMode(echoPin1, INPUT);
  pinMode(trigPin2, OUTPUT);
  pinMode(echoPin2, INPUT);
}

void loop() {
  // Measure distance for Sensor 1
  digitalWrite(trigPin1, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin1, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin1, LOW);
  duration1 = pulseIn(echoPin1, HIGH);
  distance1 = duration1 * 0.034 / 2.0;  // Calculate distance in meters

  // Measure distance for Sensor 2
  digitalWrite(trigPin2, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin2, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin2, LOW);
  duration2 = pulseIn(echoPin2, HIGH);
  distance2 = duration2 * 0.034 / 2.0;  // Calculate distance in meters

  // Send distances over serial as comma-separated values
  Serial.print(distance1);
  Serial.print(",");
  Serial.println(distance2);

  delay(1500);  // Adjust delay as needed for your application
}
