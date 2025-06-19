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
        # Release the camera if it failed to read
        if cap:
            cap.release()
        running = False
        return

    # --- Start: Display frame logic (updated) ---
    original_height, original_width, _ = frame.shape
    canvas_width = 800
    canvas_height = 500

    # Calculate new dimensions to fit within canvas while maintaining aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate dimensions based on fitting width
    new_width = canvas_width
    new_height = int(new_width / aspect_ratio)

    # If the calculated height is too large, scale based on height instead
    if new_height > canvas_height:
        new_height = canvas_height
        new_width = int(new_height * aspect_ratio)

    # Resize the frame for display on the canvas
    display_frame = cv2.resize(frame, (new_width, new_height))

    # Create a blank image the size of the canvas to center the resized frame
    final_display_image_bgr = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Calculate paste position for centering
    x_offset = (canvas_width - new_width) // 2
    y_offset = (canvas_height - new_height) // 2

    # Paste the resized frame onto the blank image
    final_display_image_bgr[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = display_frame

    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(final_display_image_bgr, cv2.COLOR_BGR2RGB)))
    canvas.imgtk = img_tk
    canvas.create_image(0, 0, anchor=NW, image=img_tk)
    # --- End: Display frame logic ---

    pred_label = predict_frame(frame) 
    label_result.config(text=pred_label)

    root.after(10, classify_and_display)

def start_camera():
    global cap, running
    if running: 
        return

    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        label_result.config(text="Gagal membuka kamera. Coba indeks kamera lain atau pastikan kamera terhubung.")
        return
    
    running = True
    classify_and_display()

def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
        cap = None # Set cap to None after releasing
    canvas.delete("all")
    label_result.config(text="Kamera dimatikan.")

# === Fungsi klasifikasi gambar upload ===
def upload_image():
    global running
    # Ensure camera is stopped if it's running
    if running:
        stop_camera()

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    image = Image.open(file_path).convert("RGB")
    
    # Resize for display on canvas while maintaining aspect ratio
    original_img_width, original_img_height = image.size
    canvas_width = 800
    canvas_height = 500

    aspect_ratio = original_img_width / original_img_height
    new_display_width = canvas_width
    new_display_height = int(new_display_width / aspect_ratio)

    if new_display_height > canvas_height:
        new_display_height = canvas_height
        new_display_width = int(new_display_height * aspect_ratio)

    display_image_pil = image.resize((new_display_width, new_display_height), Image.LANCZOS)
    
    # Create a blank image to center the resized image
    final_display_image_pil = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0)) # Black background
    x_offset = (canvas_width - new_display_width) // 2
    y_offset = (canvas_height - new_display_height) // 2
    final_display_image_pil.paste(display_image_pil, (x_offset, y_offset))

    # For model prediction, resize to 224x224
    resized_for_model = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_for_model)
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
    img_tk = ImageTk.PhotoImage(final_display_image_pil)
    canvas.imgtk = img_tk
    canvas.create_image(0, 0, anchor=NW, image=img_tk)
    label_result.config(text=label)

# === Fungsi klasifikasi frame (kamera) ===
def predict_frame(frame):
    # The model expects a 224x224 input
    img = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB for TensorFlow
    img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0) # verbose=0 to suppress prediction output
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    if confidence < threshold:
        return "Not a Cat"
    else:
        return f"{predicted_class} ({confidence:.2f}%)"

# === UI Setup ===
root = Tk()
root.title("Cat Breed Classification")
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
title_label = Label(root, text="Cat Breed Classification App", font=("Helvetica", 24, "bold"), fg="#FFFFFF", bg="#4A90E2")
title_label.pack(pady=10)

# Jalankan aplikasi
root.mainloop()