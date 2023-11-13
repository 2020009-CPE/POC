from ultralytics import YOLO
import cv2

# load yolov8 model
model = YOLO('best.pt')

# open camera
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera, you can change it if you have multiple cameras

# read frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # detect objects
    # track objects
    results = model.track(frame, persist=True)

    # plot results
    frame_ = results[0].plot()

    # visualize
    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# release the camera and close the window
cap.release()
cv2.destroyAllWindows()