# Import the OpenCV library, our main tool
import cv2

# Find your primary webcam (usually numbered 0)
cap = cv2.VideoCapture(0)

# Create a loop that runs forever to show a live video feed
while True:
    # Read one single image (a frame) from the webcam
    ret, frame = cap.read()

    # If the frame was read correctly, display it
    if ret:
        cv2.imshow("FlowGo v0.1", frame)

    # Create an exit condition: if the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanly shut down the camera and close all windows
cap.release()
cv2.destroyAllWindows()