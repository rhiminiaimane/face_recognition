import cv2
import os
import numpy as np
import pandas as pd

# Manually Enter Student Details
student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

# Directory to Save Face Vectors
base_path = r'/train'
student_folder = os.path.join(base_path, f"{student_id}_{student_name}")

if not os.path.exists(student_folder):
    os.makedirs(student_folder)

# Load Haar Cascade for Face Detection
haar_cascade = cv2.CascadeClassifier(r'C:\Users\DELL\Desktop\project\face_recognition\haarcascade_frontalface_default.xml')

# Open Webcam
cap = cv2.VideoCapture(0)
count = 0
max_images = 100  # Number of images to capture

# Excel File to Store Metadata
excel_file = os.path.join(base_path, "student_data.xlsx")
data = []

while count < max_images:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  # Extract face region
        face_resized = cv2.resize(face_roi, (100, 100))  # Resize for uniformity
        
        # Convert face to NumPy vector
        face_vector = face_resized.flatten()

        # Save as .npy file (NumPy format)
        vector_filename = f"{student_name}_{count}.npy"
        vector_path = os.path.join(student_folder, vector_filename)
        np.save(vector_path, face_vector)

        # Store metadata in Excel
        data.append([student_id, student_name, vector_path])

        print(f"Saved {vector_path}")
        count += 1

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Capturing Face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save Metadata to Excel
df = pd.DataFrame(data, columns=["Student ID", "Student Name", "Vector Path"])
if os.path.exists(excel_file):
    df_existing = pd.read_excel(excel_file)
    df = pd.concat([df_existing, df], ignore_index=True)

df.to_excel(excel_file, index=False)

print(f"Student data saved in {excel_file}")
