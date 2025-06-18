import tensorflow as tf
import numpy as np
import cv2
import json
import os
from tkinter import *
from PIL import Image, ImageTk

# === Load model dan class indices ===
model_path = 'cat_breed_classifier_final_mobilenet.keras'
class_indices_path = 'class_indices.json'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
if not os.path.exists(class_indices_path):
    raise FileNotFoundError(f"File class_indices tidak ditemukan: {class_indices_path}")

model = tf.keras.models.load_model(model_path)
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for class_name, index in class_indices.items():
    class_names[index] = class_name

# === Variabel global ===
cap = None
running = False
threshold = 10  # Minimum confidence %

# === Fungsi klasifikasi dan kamera ===
def classify_and_display():
    global cap, running
    if not running:
        return
    
    ret, frame = cap.read()
    if not ret:
        label_result.config(text="Tidak bisa mengambil frame.")
        return

    img = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    if confidence < threshold:
        label = "Bukan Kucing"
    else:
        label = f"{predicted_class} ({confidence:.2f}%)"

    label_result.config(text=label)

    # Tampilkan frame di canvas
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    canvas.imgtk = img_tk
    canvas.create_image(0, 0, anchor=NW, image=img_tk)

    root.after(10, classify_and_display)

def start_camera():
    global cap, running
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        label_result.config(text="Gagal membuka kamera.")
        return
    running = True
    classify_and_display()

def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
    canvas.delete("all")
    label_result.config(text="Kamera dimatikan.")

# === UI Setup ===
root = Tk()
root.title("Aplikasi Klasifikasi Ras Kucing")
root.geometry("900x650")  # Window lebar

# ==== Frame tampilan kamera ====
canvas_frame = Frame(root)
canvas_frame.pack(pady=10)

canvas = Canvas(canvas_frame, width=800, height=500, bg='black')
canvas.pack()

# ==== Label hasil prediksi ====
label_result = Label(root, text="Tekan Start untuk klasifikasi", font=("Helvetica", 18), fg="blue")
label_result.pack(pady=15)

# ==== Tombol kontrol ====
button_frame = Frame(root)
button_frame.pack(pady=10)

btn_start = Button(button_frame, text="Start Camera", command=start_camera, width=20, height=2, font=("Arial", 14),
                   bg='green', fg='white')
btn_start.grid(row=0, column=0, padx=20)

btn_stop = Button(button_frame, text="Stop", command=stop_camera, width=20, height=2, font=("Arial", 14),
                  bg='red', fg='white')
btn_stop.grid(row=0, column=1, padx=20)

# === Jalankan aplikasi ===
root.mainloop()
