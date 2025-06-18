import tensorflow as tf
import numpy as np
import cv2
import json
import os
from tkinter import *
from tkinter import filedialog
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

# === Fungsi klasifikasi dari kamera ===
def classify_and_display():
    global cap, running
    if not running:
        return
    
    ret, frame = cap.read()
    if not ret:
        label_result.config(text="Tidak bisa mengambil frame.")
        return

    pred_label = predict_frame(frame)
    label_result.config(text=pred_label)

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

# === Fungsi klasifikasi gambar upload ===
def upload_image():
    global running
    running = False
    if cap:
        cap.release()

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    image = Image.open(file_path).convert("RGB")
    resized_image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
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

    # Tampilkan ke canvas
    display_image = image.resize((800, 500))
    img_tk = ImageTk.PhotoImage(display_image)
    canvas.imgtk = img_tk
    canvas.create_image(0, 0, anchor=NW, image=img_tk)
    label_result.config(text=label)

# === Fungsi klasifikasi frame (kamera) ===
def predict_frame(frame):
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
        return "Bukan Kucing"
    else:
        return f"{predicted_class} ({confidence:.2f}%)"

# === UI Setup ===
root = Tk()
root.title("Klasifikasi Ras Kucing")
root.geometry("1000x700")
root.configure(bg="#4A90E2")

# Main frame
main_frame = Frame(root, bg="#FFFFFF", padx=20, pady=20, relief="raised", bd=5)
main_frame.place(relx=0.5, rely=0.5, anchor=CENTER)

# Canvas frame
canvas_frame = Frame(main_frame, bg="#FFFFFF")
canvas_frame.pack(pady=10)

canvas = Canvas(canvas_frame, width=800, height=500, bg='#E6F0FA', highlightthickness=2, highlightbackground="#4A90E2")
canvas.pack()

# Result label
label_result = Label(main_frame, text="Tekan Start atau Upload Gambar", font=("Helvetica", 20, "bold"), fg="#2E2E2E", bg="#FFFFFF")
label_result.pack(pady=20)

# Button frame
button_frame = Frame(main_frame, bg="#FFFFFF")
button_frame.pack(pady=20)

btn_start = Button(button_frame, text="Start Camera", command=start_camera, width=15, height=2, font=("Arial", 14, "bold"),
                   bg="#4CAF50", fg="white", activebackground="#45a049", relief="flat", cursor="hand2")
btn_start.grid(row=0, column=0, padx=10)

btn_stop = Button(button_frame, text="Stop", command=stop_camera, width=15, height=2, font=("Arial", 14, "bold"),
                  bg="#F44336", fg="white", activebackground="#da190b", relief="flat", cursor="hand2")
btn_stop.grid(row=0, column=1, padx=10)

btn_upload = Button(button_frame, text="Upload Gambar", command=upload_image, width=15, height=2, font=("Arial", 14, "bold"),
                    bg="#2196F3", fg="white", activebackground="#1976D2", relief="flat", cursor="hand2")
btn_upload.grid(row=0, column=2, padx=10)

# Decorative title
title_label = Label(root, text="Aplikasi Klasifikasi Ras Kucing", font=("Helvetica", 24, "bold"), fg="#FFFFFF", bg="#4A90E2")
title_label.pack(pady=10)

# Jalankan aplikasi
root.mainloop()