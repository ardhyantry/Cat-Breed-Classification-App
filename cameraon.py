import tensorflow as tf
import json
import numpy as np
import cv2  
import os

# Memuat model yang telah dilatih
model_path = 'cat_breed_classifier_final_mobilenet.keras'
class_indices_path = 'class_indices.json'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di path: {model_path}")

model = tf.keras.models.load_model(model_path)
print("Model berhasil dimuat.")

if not os.path.exists(class_indices_path):
    raise FileNotFoundError(f"File class_indices tidak ditemukan di path: {class_indices_path}")

with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Membuat daftar class_names berdasarkan indeks
class_names = [None] * len(class_indices)
for class_name, index in class_indices.items():
    class_names[index] = class_name

print("class_indices berhasil dimuat dan class_names dibuat.")

# Menginisialisasi kamera (0 adalah indeks default kamera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise IOError("Tidak dapat membuka kamera.")

print("Mengakses kamera. Tekan 'q' untuk keluar.")

# Threshold untuk mendeteksi bukan kucing (misalnya 50%)
threshold = 10

# Optimize: Load face cascade once outside the loop to improve performance
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

# Optimize: Add frame skipping for prediction to reduce CPU usage
frame_skip = 3  # Process every 3rd frame for prediction
frame_count = 0
last_label = "Initializing..."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame dari kamera.")
        break
    
    frame_count += 1
    
    # Optimize: Use pre-loaded face cascade instead of loading it every frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Optimize: Removed excessive print statements that slow down the loop
    # Draw bounding box if cat face detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Optimize: Only run prediction every N frames to reduce CPU usage
    if frame_count % frame_skip == 0:
        # Mengubah ukuran gambar untuk prediksi
        img = cv2.resize(frame, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Mengubah gambar menjadi array
        img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
        
        # Menambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Melakukan preprocessing sesuai MobileNet
        img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
        
        # Membuat prediksi
        predictions = model.predict(img_array, verbose=0)  # Optimize: verbose=0 to suppress output
        score = tf.nn.softmax(predictions[0])
        
        # Mendapatkan nama kelas dengan probabilitas tertinggi
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        # Jika confidence lebih rendah dari threshold, maka dianggap bukan kucing
        if confidence < threshold:
            last_label = "Bukan Kucing: Tidak cukup yakin"
        else:
            last_label = f"{predicted_class}: {confidence:.2f}%"
    
    # Use cached label for non-prediction frames
    label = last_label
    
    # Menentukan posisi teks
    position = (10, 30)
    # Menambahkan teks ke frame
    cv2.putText(frame, label, position, 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Menampilkan frame dengan bounding box
    cv2.imshow('Cat Breed Classification with Bounding Box', frame)
    
    # Menunggu input key, jika 'q' ditekan maka keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Keluar dari program.")
        break

# Melepaskan kamera dan menutup semua jendela
cap.release()
cv2.destroyAllWindows()
