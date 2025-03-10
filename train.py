import os
import cv2 as cv
import numpy as np

# Load Haar Cascade for Face Detection
haar_cascade = cv.CascadeClassifier(r'C:\Users\HP\Desktop\Projets\Deep Learning\face_recognition\haarcascade_frontalface_default.xml')

# Dataset Directory
DATASET_DIR = r'train'

# Face Recognizer Model
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Training Data Storage
features = []
labels = []

# Function to Load Face Vectors & Apply Haar Cascade
def load_data():
    global features, labels
    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)

        # Ensure the folder name follows 'ID_Name' format
        if "_" not in folder:
            print(f" Skipping folder with incorrect format: {folder}")
            continue

        student_info = folder.split("_", 1)

        try:
            student_id = int(student_info[0])  # Extract Student ID (must be an integer)
            student_name = student_info[1]
        except ValueError:
            print(f" Skipping invalid folder: {folder} (ID must be a number)")
            continue

        print(f"Processing Student: {student_name} (ID: {student_id})")

        # Iterate through `.npy` face vectors
        for file in os.listdir(folder_path):
            if file.endswith('.npy'):
                file_path = os.path.join(folder_path, file)

                # Load face vector
                face_vector = np.load(file_path)
                face_image = face_vector.reshape(100, 100)  # Reshape to original size

                # Convert to grayscale (ensuring compatibility with Haar Cascade)
                gray_face = np.uint8(face_image)

                # Apply Haar Cascade Face Detection
                faces_rects = haar_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=4)

                if len(faces_rects) > 0:
                    for (x, y, w, h) in faces_rects:
                        face_roi = gray_face[y:y+h, x:x+w]
                        features.append(face_roi)
                        labels.append(student_id)
                        print(f"Face Detected: {file_path} (ID: {student_id})")
                else:
                    print(f"No Face Detected in {file_path}")

# Run Data Loading
load_data()
print("Training Data Loaded!")

# Convert Lists to NumPy Arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Train the Model
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

print("Training Completed and Model Saved!")
