import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        # This new line draws the green rectangle
        cv2.rectangle(frame, (100, 80), (540, 400), (0, 255, 0), 2)

        cv2.imshow("FlowGo v0.1", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()