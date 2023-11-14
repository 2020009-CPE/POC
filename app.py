import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO
import mysql.connector

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('peopledetect.pt')

# Initialize MySQL database connection
conn = mysql.connector.connect(
    host='localhost',  # Replace with your MySQL server host
    user='root',  # Replace with your MySQL username
    password='',  # Replace with your MySQL password
    database='peoplecount'  # Replace with your database name
)
cursor = conn.cursor()

# Create a table to store the counts if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS people_counts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    person_in INT,
                    person_out INT,
                    total_count INT
                )''')

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(3)  # Change camera index if needed
        self.people_counter = PeopleCounter()

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Detect and count people
        results = model.track(frame, persist=True)
        count_in, count_out, total_count = self.people_counter.count_people(results)

        # Visualize bounding boxes
        frame_ = results[0].plot()

        # Add the counts below the bounding boxes with white text
        text = f"Total: {total_count}, In: {count_in}, Out: {count_out}"
        cv2.putText(frame_, text, (10, frame_.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White color

        # Insert counts into the database
        self.people_counter.insert_counts_to_db()

        ret, jpeg = cv2.imencode('.jpg', frame_)
        return jpeg.tobytes()

class PeopleCounter:
    def __init__(self):
        self.count_in = 0
        self.count_out = 0
        self.total_count = 0

    def count_people(self, results):
        self.count_in = 0
        self.count_out = 0

        for box in results[0].boxes:
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            class_id = results[0].names[box.cls[0].item()]

            # Extract the color based on class_id
            color = (0, 0, 0)  # Default color (black)

            if class_id == "PEOPLEIN":
                color = (0, 165, 255)  # Orange for PEOPLEIN
                self.count_in += 1
            elif class_id == "PEOPLEOUT":
                color = (0, 255, 255)  # Yellow for PEOPLEOUT
                self.count_out += 1

        self.total_count = self.count_in + self.count_out  # Use addition here
        return self.count_in, self.count_out, self.total_count

    def insert_counts_to_db(self):
        cursor.execute('''INSERT INTO people_counts (person_in, person_out, total_count)
                          VALUES (%s, %s, %s)''', (self.count_in, self.count_out, self.total_count))
        conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
