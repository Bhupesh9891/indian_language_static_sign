import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
import glob
from collections import deque
import tkinter as tk
from tkinter import ttk

from isl_features import extract_features_from_landmarks, HAND_CONNECTIONS

MODEL_FILE = "my_hand_model.p"
NEUTRAL_SIGN_NAME = "Custom_Neutral"

if not os.path.exists(MODEL_FILE):
    print("❌ Model file not found! Please train a model first.")
    exit(1)

def detect_available_cameras():
    cameras = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_file = os.path.join(path, "name")
        if os.path.exists(name_file):
            try:
                with open(name_file, "r") as f: name = f.read().strip()
                dev_node = os.path.basename(path)
                idx = int(dev_node.replace("video", ""))
                cameras.append((idx, f"[{idx}] {name}"))
            except Exception: pass

    if not cameras:
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((i, f"Camera Index {i}"))
                cap.release()

    return cameras if cameras else [(0, "Default Camera (Index 0)")]


def ask_camera_popup():
    cameras = detect_available_cameras()
    selected_index = [cameras[0][0]]

    root = tk.Tk()
    root.title("Live Test - Select Camera")
    root.geometry("380x180")
    root.resizable(False, False)
    root.eval('tk::PlaceWindow . center')

    tk.Label(root, text="🤟 Indian Sign Language Tester", font=("Segoe UI", 12, "bold")).pack(pady=10)
    tk.Label(root, text="Select your camera:", font=("Segoe UI", 10)).pack(pady=2)

    camera_labels = [c[1] for c in cameras]
    combo = ttk.Combobox(root, values=camera_labels, state="readonly", width=35)
    combo.current(0)
    combo.pack(pady=10)

    def on_confirm():
        chosen = combo.get()
        for idx, label in cameras:
            if label == chosen:
                selected_index[0] = idx
                break
        root.destroy()

    tk.Button(root, text="Launch Camera", command=on_confirm, bg="#4CAF50", fg="white",
              font=("Segoe UI", 10, "bold"), padx=15, pady=4).pack(pady=5)
    root.mainloop()
    return selected_index[0]


chosen_cam = ask_camera_popup()
camera = cv2.VideoCapture(chosen_cam)
if not camera.isOpened():
    print(f"❌ Could not open camera {chosen_cam}!")
    exit(1)

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# MediaPipe Setup
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.35,
    min_tracking_confidence=0.35
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

last_timestamp = 0
prediction_buffer = deque(maxlen=6)
print("🚀 Live Test Running! Press 'q' on the video window to stop.")

while True:
    success, frame = camera.read()
    if not success: continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Glassy Top Header Bar
    header_overlay = frame.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    cv2.addWeighted(header_overlay, 0.75, frame, 0.25, 0, frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    current_timestamp = int(time.time() * 1000)
    if current_timestamp <= last_timestamp: current_timestamp = last_timestamp + 1
    last_timestamp = current_timestamp

    result = landmarker.detect_for_video(mp_image, current_timestamp)

    header_text = "Waiting for hands..."
    status_color = (160, 160, 160)
    bone_color = (220, 200, 50)

    if result.hand_landmarks:
        features = extract_features_from_landmarks(result.hand_landmarks)
        if features:
            probs = model.predict_proba([np.asarray(features, dtype=np.float32)])[0]
            classes = model.classes_
            sorted_indices = np.argsort(probs)[::-1]

            top_idx = sorted_indices[0]
            top_pred = classes[top_idx]
            top_conf = int(probs[top_idx] * 100)

            margin = (probs[top_idx] - probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 1.0

            if top_conf >= 75 and margin >= 0.12:
                prediction_buffer.append(top_pred)
            else:
                prediction_buffer.append("Uncertain")

            counts = {}
            for p in prediction_buffer:
                counts[p] = counts.get(p, 0) + 1
            smooth_pred = max(counts, key=counts.get)

            if smooth_pred == NEUTRAL_SIGN_NAME:
                status_color = (180, 180, 180)
                bone_color = (180, 180, 180)
                header_text = f"Neutral / Idle Hand ({top_conf}%)"
            elif smooth_pred != "Uncertain":
                status_color = (0, 220, 100)
                bone_color = (0, 220, 100)
                header_text = f"Sign: {smooth_pred}   |   Confidence: {top_conf}%"
            else:
                status_color = (0, 215, 255)
                bone_color = (220, 200, 50)
                header_text = f"Detecting... ({top_conf}%)"
    else:
        prediction_buffer.clear()

    # Draw Skeletons
    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, pts[start_idx], pts[end_idx], bone_color, 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)

    cv2.putText(frame, header_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
    cv2.imshow("Indian Sign Language Live Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()