import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8n model (the "brain")
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Tell the YOLO model to find objects in the frame
        results = model(frame, stream=True)

        # Loop through the results from the model
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Get the coordinates of the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to whole numbers

                # Draw a rectangle around the detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Get the name of the object
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Put the object name on the screen
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)


        # Display the frame with detections
        cv2.imshow("FlowGo v0.1", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()