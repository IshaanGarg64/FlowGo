import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

zone_x1, zone_y1, zone_x2, zone_y2 = 100, 80, 540, 400

virtual_cart = []
items_in_zone = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    zone_color = (0, 255, 0) # Green
    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if class_name == 'bottle':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check if the bottle's center is inside the detection zone
                if zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
                    zone_color = (0, 0, 255) # Red
                    
                    if class_name not in items_in_zone:
                        virtual_cart.append(class_name)
                        items_in_zone.add(class_name)
                        print(f"Item Added: {class_name} | Cart: {virtual_cart}")
                
                # --- THIS LOGIC IS NOW ACTIVE ---
                else:
                    # If the bottle is outside the zone, check if we remember it being inside
                    if class_name in items_in_zone:
                        # If so, remove it from the cart and our memory
                        virtual_cart.remove(class_name)
                        items_in_zone.remove(class_name)
                        print(f"Item Removed: {class_name} | Cart: {virtual_cart}")
                # --- END OF ACTIVE LOGIC ---
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), zone_color, 2)

    cart_text = f"Cart: {virtual_cart}"
    cv2.putText(frame, cart_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("FlowGo v0.1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()