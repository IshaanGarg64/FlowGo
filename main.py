import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the coordinates for our "detection zone"
zone_x1, zone_y1, zone_x2, zone_y2 = 100, 80, 540, 400

# Loop through the video frames
while True:
    ret, frame = cap.read()

    if ret:
        # Draw the detection zone on the frame
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 0), 2)

        # Tell YOLO to find objects in the frame
        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                # Get the object's class name
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # --- NEW LOGIC STARTS HERE ---
                # Check if the detected object is a "bottle"
                if class_name == 'bottle':
                    # Get the coordinates of the bottle
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Calculate the center of the bottle's box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Draw a box and a label for the bottle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

                    # Check if the bottle's center is inside the detection zone
                    if center_x > zone_x1 and center_x < zone_x2 and center_y > zone_y1 and center_y < zone_y2:
                        # If it is, print a message to the terminal
                        print("Bottle detected inside the zone!")
                # --- NEW LOGIC ENDS HERE ---

        cv2.imshow("FlowGo v0.1", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()