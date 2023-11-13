from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load YOLO model
model = YOLO('best.pt')

# Open camera
cap = cv2.VideoCapture(1)
show_live_camera = True  # Flag to toggle between live camera and uploaded content

def generate_frames():
    while show_live_camera:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.track(frame, persist=True)
            frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if show_live_camera:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
