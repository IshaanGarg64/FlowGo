import cv2
from ultralytics import YOLO

# --- UPGRADE 1: Using a more accurate model ---
# We are now using 'yolov8s.pt' (small) instead of 'yolov8n.pt' (nano)
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the detection zone coordinates
zone_x1, zone_y1, zone_x2, zone_y2 = 100, 80, 540, 400

# This is our master "memory" of which item IDs are in the zone
items_in_zone = set()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get results from the AI model with tracking
    # 'persist=True' helps the tracker remember objects between frames
    # 'verbose=False' hides extra YOLO output in the terminal
    results = model.track(frame, persist=True, verbose=False)

    # This dictionary will temporarily hold the name for each tracked ID in the current frame
    tracked_items = {}
    if results[0].boxes.id is not None:
        # Get all the tracking data
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy() # Get confidence scores

        # Loop through all detected objects in this frame
        for box, track_id, class_id, conf in zip(boxes, ids, class_ids, confidences):
            
            # --- UPGRADE 2: Filtering by confidence score ---
            # We will only process detections with a confidence > 50% (0.5)
            if conf > 0.5:
                x1, y1, x2, y2 = box
                class_name = model.names[class_id]
                tracked_items[track_id] = class_name # Store the name for this ID

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Check if the object is inside the detection zone
                if zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
                    # If it is, add its unique ID to our memory set
                    items_in_zone.add(track_id)
                else:
                    # If it's outside the zone, remove it from our memory set
                    items_in_zone.discard(track_id)
                
                # Draw the box and label for every high-confidence detected object
                label = f"ID: {track_id} {class_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # Rebuild the virtual cart from scratch on every frame for accuracy
    virtual_cart = []
    for track_id in items_in_zone:
        if track_id in tracked_items:
            virtual_cart.append(tracked_items[track_id])

    # Set the zone color based on whether any item is inside
    if items_in_zone:
        zone_color = (0, 0, 255) # Red
    else:
        zone_color = (0, 255, 0) # Green

    # Draw the zone and the cart info on the screen
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), zone_color, 2)
    cart_text = f"Cart: {virtual_cart}"
    cv2.putText(frame, cart_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the final frame
    cv2.imshow("FlowGo v0.1", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()