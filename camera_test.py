import cv2

# URL of the phone camera
# Replace with the correct URL provided by the IP Webcam or DroidCam app
stream_url = 'http://100.102.232.63:4747/video'

# Open the connection to the camera
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Unable to connect to the camera. Please check the URL and network connection.")
    exit()

print("Successfully connected to the camera. Press 'q' to exit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Unable to read frame. Exiting the program.")
        break

    # Rotate the frame 90 degrees counterclockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Display the frame
    cv2.imshow('Test Camera', frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources and close the windows
cap.release()
cv2.destroyAllWindows()