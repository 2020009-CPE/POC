import cv2
from ultralytics import YOLO
import mysql.connector
from time import sleep

# Load YOLOv8 model
model = YOLO('peopledetect.pt')

# Open camera
cap = cv2.VideoCapture(3)  # 0 corresponds to the default camera, you can change it if you have multiple cameras

# Increase FPS
fps = 120  # Set a higher value, e.g., 60 FPS

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

            # Draw bounding box with color
            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color, 2)

        self.total_count = self.count_in + self.count_out  # Use addition here
        return self.count_in, self.count_out, self.total_count

    def insert_counts_to_db(self):
        cursor.execute('''INSERT INTO people_counts (person_in, person_out, total_count)
                          VALUES (%s, %s, %s)''', (self.count_in, self.count_out, self.total_count))
        conn.commit()

# Create PeopleCounter instance
people_counter = PeopleCounter()

# Read frames and detect/count people
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect and count people
    results = model.track(frame, persist=True)

    # Update and display the counts
    count_in, count_out, total_count = people_counter.count_people(results)

    # Visualize bounding boxes
    frame_ = results[0].plot()

    # Add the counts below the bounding boxes with white text
    text = f"Total: {total_count}, In: {count_in}, Out: {count_out}"
    cv2.putText(frame_, text, (10, frame_.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White color

    # Visualize
    cv2.imshow('frame', frame_)

    # Insert counts into the database
    people_counter.insert_counts_to_db()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
