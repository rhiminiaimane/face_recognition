# Face Recognition System for Student Identification

A computer vision project built using **OpenCV** and **LBPH Face Recognizer** to **capture, train, and recognize** student faces using webcam input. This tool allows institutions to register and recognize students through facial features efficiently.

---

## 📂 Project Structure

```bash
face_recognition_project/
├── train/                  # Folder storing captured student face vectors
│   ├── 101_John/          # Sample student folder with .npy face vectors
│   └── 102_Alice/
├── haarcascade_frontalface_default.xml
├── student_data.xlsx      # Excel file storing metadata of registered students
├── face_trained.yml       # Trained LBPH face recognizer model
├── features.npy           # Saved features used for training
├── labels.npy             # Saved labels used for training
├── capture_faces.py       # Script to capture student faces and store vectors
├── train_model.py         # Script to load vectors and train face recognizer
├── recognize_faces.py     # Script to recognize students in real-time
└── README.md              # Project documentation
```

---

## 📌 Features

* 📸 **Face Capture**: Captures 100 grayscale face images per student.
* 🧠 **Model Training**: Trains an LBPH model using saved `.npy` vectors.
* 🧾 **Metadata Logging**: Automatically logs student ID, name, and image path into an Excel file.
* 🔍 **Real-Time Recognition**: Recognizes faces using webcam and displays identity with confidence score.

---

## 🔧 Requirements

* Python 3.12
* OpenCV (`opencv-python`, `opencv-contrib-python`)
* NumPy
* Pandas

Install them using:

```bash
pip install opencv-python opencv-contrib-python numpy pandas
```

---

## 🎬 How It Works

### 1. 📷 Capture Student Faces

Run the script:

```bash
python capture_faces.py
```

You’ll be prompted to enter:

* Student ID
* Student Name

The system then opens the webcam, detects faces, saves them as `.npy` files, and logs metadata in `student_data.xlsx`.

---

### 2. 🏋️ Train the Face Recognition Model

Run the training script:

```bash
python train_model.py
```

It loads all `.npy` files, reshapes them to images, re-validates with Haar Cascade, and trains the model. Outputs:

* `face_trained.yml`
* `features.npy`
* `labels.npy`

---

### 3. 🧑‍💻 Real-Time Face Recognition

To recognize faces live from webcam:

```bash
python recognize_faces.py
```

This uses the trained model and matches detected faces with known students. It displays:

* **Student Name**
* **Confidence Score**
* Color-coded bounding box (Green = recognized, Red = unknown)

Press **`q`** to exit the webcam stream.

---

## ⚠️ Notes

* Haar cascade XML file must exist at the specified path in all scripts:

  ```
  C:\Users\HP\Desktop\Projets\Deep Learning\face_recognition\haarcascade_frontalface_default.xml
  ```

  Change this path as needed.

* Make sure all student folders are named in the format:
  **`<ID>_<Name>`** (e.g., `101_John`, `102_Alice`).

* Face images are resized to **100x100** for consistent training.

---

## 📈 Future Improvements

* Add GUI for data entry and visualization
* Improve recognition accuracy with deep learning models (e.g., FaceNet, Dlib)
* Add attendance logging
* Implement facial recognition via smartphone camera

---
