import cv2 as cv
import numpy as np
import os

# Load Haar Cascade for Face Detection
haar_cascade = cv.CascadeClassifier(r'C:\Users\HP\Desktop\Projets\Deep Learning\face_recognition\haarcascade_frontalface_default.xml')

# Load Trained Face Recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Load Labels (Student IDs)
labels = np.load('labels.npy')

# Map Student IDs to Names (Handle Naming Errors)
STUDENT_DIR = r'train'
id_to_name = {}

for folder in os.listdir(STUDENT_DIR):
    folder_path = os.path.join(STUDENT_DIR, folder)

    # Ensure folder follows "ID_Name" format
    if "_" not in folder:
        print(f" Skipping invalid folder: {folder} (No underscore '_')")
        continue

    student_info = folder.split("_", 1)

    # Check if the first part is a valid number
    if not student_info[0].isdigit():
        print(f" Skipping folder with non-numeric ID: {folder}")
        continue

    student_id = int(student_info[0])
    student_name = student_info[1]

    id_to_name[student_id] = student_name

print(f" Loaded Student Names: {id_to_name}")

# Open Webcam
cap = cv.VideoCapture('http://192.168.0.120:4747/video') 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rects:
        face_roi = gray[y:y+h, x:x+w]

        # Recognize Face
        label, confidence = face_recognizer.predict(face_roi)

        # Get Name from ID
        student_name = id_to_name.get(label, "Unknown")

        # Draw Rectangle and Label
        color = (0, 255, 0) if student_name != "Unknown" else (0, 0, 255)
        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv.putText(frame, f"{student_name} ({confidence:.2f})", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        print(f"Recognized: {student_name} (Confidence: {confidence:.2f})")

    # Display Webcam Feed
    cv.imshow("Live Face Recognition", frame)

    # Press 'q' to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
