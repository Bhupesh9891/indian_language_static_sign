import os
import sys

# Silence C++ logs before importing libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import glob
import difflib
import subprocess
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from isl_features import extract_features_from_landmarks, HAND_CONNECTIONS

DATASET_DIR = "dataset"
MODEL_FILE = "my_hand_model.p"
NEUTRAL_SIGN_NAME = "Custom_Neutral"
os.makedirs(DATASET_DIR, exist_ok=True)

# -------------------------------------------------------------
# Zero-Lag Window Manager & Safe Popups (Flushes Linux X11/Wayland)
# -------------------------------------------------------------
_TK_MASTER = None

def get_tk_master():
    global _TK_MASTER
    if _TK_MASTER is None:
        _TK_MASTER = tk.Tk()
        _TK_MASTER.withdraw()
    return _TK_MASTER

def close_dialog_cleanly(win):
    """Flushes Tkinter buffers instantly so custom dialogs never freeze or linger."""
    master = get_tk_master()
    win.destroy()
    master.update_idletasks()
    master.update()

def safe_ask_yes_no(title, msg):
    """Native yes/no popup with immediate Linux display buffer flush."""
    master = get_tk_master()
    ans = messagebox.askyesno(title, msg, parent=master)
    master.update_idletasks()
    master.update()
    return ans

def safe_show_info(title, msg):
    """Native info popup with immediate Linux display buffer flush."""
    master = get_tk_master()
    messagebox.showinfo(title, msg, parent=master)
    master.update_idletasks()
    master.update()

def safe_show_warning(title, msg):
    """Native warning popup with immediate Linux display buffer flush."""
    master = get_tk_master()
    messagebox.showwarning(title, msg, parent=master)
    master.update_idletasks()
    master.update()

def center_window(win, width, height):
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = max(0, (sw - width) // 2)
    y = max(0, (sh - height) // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")


# -------------------------------------------------------------
# 1. Camera Detection & Launcher
# -------------------------------------------------------------
def detect_available_cameras():
    cameras = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_file = os.path.join(path, "name")
        if os.path.exists(name_file):
            try:
                with open(name_file, "r") as f:
                    name = f.read().strip()
                dev_node = os.path.basename(path)
                idx = int(dev_node.replace("video", ""))
                cameras.append((idx, f"[{idx}] {name}"))
            except Exception:
                pass

    if not cameras:
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((i, f"Camera Index {i}"))
                cap.release()

    return cameras if cameras else [(0, "Default Camera (Index 0)")]


def ask_startup_popup():
    cameras = detect_available_cameras()
    selected_camera = [cameras[0][0]]
    selected_mode = ["record"]

    master = get_tk_master()
    win = tk.Toplevel(master)
    win.title("ISL Studio - Launcher")
    win.resizable(False, False)
    center_window(win, 440, 260)

    tk.Label(win, text="🤟 Indian Sign Language Studio", font=("Segoe UI", 12, "bold")).pack(pady=10)
    tk.Label(win, text="Select your camera:", font=("Segoe UI", 10)).pack(pady=2)

    camera_labels = [c[1] for c in cameras]
    combo = ttk.Combobox(win, values=camera_labels, state="readonly", width=38)
    combo.current(0)
    combo.pack(pady=8)

    def resolve_chosen_cam():
        chosen_label = combo.get()
        for idx, label in cameras:
            if label == chosen_label:
                return idx
        return 0

    def on_record():
        selected_camera[0] = resolve_chosen_cam()
        selected_mode[0] = "record"
        close_dialog_cleanly(win)

    def on_test():
        selected_camera[0] = resolve_chosen_cam()
        selected_mode[0] = "test"
        close_dialog_cleanly(win)

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=15)

    tk.Button(btn_frame, text="🎥 Record / Add Signs", command=on_record, bg="#4CAF50", fg="white",
              font=("Segoe UI", 10, "bold"), padx=12, pady=6).pack(side=tk.LEFT, padx=8)
    tk.Button(btn_frame, text="🚀 Jump to Live Test", command=on_test, bg="#2196F3", fg="white",
              font=("Segoe UI", 10, "bold"), padx=12, pady=6).pack(side=tk.LEFT, padx=8)

    win.wait_window()
    return selected_camera[0], selected_mode[0]


# -------------------------------------------------------------
# 2. Standardized Sign Name Dialog (With Append Mode)
# -------------------------------------------------------------
def get_standardized_sign_dialog():
    master = get_tk_master()
    win = tk.Toplevel(master)
    win.title("Configure Sign")
    win.resizable(False, False)
    center_window(win, 450, 270)

    tk.Label(win, text="Configure Sign Label", font=("Segoe UI", 12, "bold")).pack(pady=10)

    frame_cat = tk.Frame(win)
    frame_cat.pack(pady=5)
    tk.Label(frame_cat, text="Category: ", font=("Segoe UI", 10)).pack(side=tk.LEFT)
    category_combo = ttk.Combobox(frame_cat, values=["Alphabet (A-Z)", "Number (0-9)", "Custom Word"],
                                  state="readonly", width=20)
    category_combo.current(0)
    category_combo.pack(side=tk.LEFT)

    frame_val = tk.Frame(win)
    frame_val.pack(pady=8)
    val_label = tk.Label(frame_val, text="Letter: ", font=("Segoe UI", 10))
    val_label.pack(side=tk.LEFT)

    alphabet_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    numbers_list = [str(i) for i in range(10)]

    val_combo = ttk.Combobox(frame_val, values=alphabet_list, state="readonly", width=18)
    val_combo.current(0)
    val_combo.pack(side=tk.LEFT)

    custom_entry = tk.Entry(frame_val, font=("Segoe UI", 10), width=20)
    lbl_preview = tk.Label(win, text="Label: Alphabet_A", font=("Segoe UI", 9, "italic"), fg="#4CAF50")
    lbl_preview.pack(pady=5)

    def update_preview(*args):
        cat = category_combo.get()
        if cat == "Alphabet (A-Z)":
            val_label.config(text="Letter: ")
            custom_entry.pack_forget()
            val_combo.config(values=alphabet_list)
            val_combo.pack(side=tk.LEFT)
            if val_combo.get() not in alphabet_list: val_combo.current(0)
            lbl_preview.config(text=f"Label: Alphabet_{val_combo.get()}")
        elif cat == "Number (0-9)":
            val_label.config(text="Number: ")
            custom_entry.pack_forget()
            val_combo.config(values=numbers_list)
            val_combo.pack(side=tk.LEFT)
            if val_combo.get() not in numbers_list: val_combo.current(0)
            lbl_preview.config(text=f"Label: Number_{val_combo.get()}")
        else:
            val_label.config(text="Word: ")
            val_combo.pack_forget()
            custom_entry.pack(side=tk.LEFT)
            word = custom_entry.get().strip() or "YourWord"
            lbl_preview.config(text=f"Label: Custom_{word}")

    category_combo.bind("<<ComboboxSelected>>", update_preview)
    val_combo.bind("<<ComboboxSelected>>", update_preview)
    custom_entry.bind("<KeyRelease>", update_preview)

    final_name = [None]
    append_mode = [False]

    def on_confirm():
        cat = category_combo.get()
        if cat == "Alphabet (A-Z)":
            name = f"Alphabet_{val_combo.get()}"
        elif cat == "Number (0-9)":
            name = f"Number_{val_combo.get()}"
        else:
            raw_word = custom_entry.get().strip()
            if len(raw_word) == 1 and raw_word.isalpha():
                category_combo.current(0)
                update_preview()
                val_combo.set(raw_word.upper())
                return
            if len(raw_word) == 1 and raw_word.isdigit():
                category_combo.current(1)
                update_preview()
                val_combo.set(raw_word)
                return

            clean_word = raw_word.replace("Alphabet_", "").replace("Number_", "").replace(" ", "_").capitalize()
            if not clean_word:
                messagebox.showerror("Error", "Please type a word!", parent=win)
                return

            existing = [os.path.basename(f).replace("Custom_", "").replace(".pickle", "")
                        for f in glob.glob(os.path.join(DATASET_DIR, "Custom_*.pickle"))]
            matches = difflib.get_close_matches(clean_word, existing, n=1, cutoff=0.75)
            if matches and matches[0].lower() != clean_word.lower():
                if messagebox.askyesno("Spell Check", f"Did you mean 'Custom_{matches[0]}'?", parent=win):
                    clean_word = matches[0]

            name = f"Custom_{clean_word}"

        target_file = os.path.join(DATASET_DIR, f"{name}.pickle")
        if os.path.exists(target_file):
            resp = messagebox.askyesnocancel(
                "Sign Already Exists",
                f"'{name}' already exists in your dataset!\n\n"
                f"• YES: APPEND more samples (adds natural human variation)\n"
                f"• NO: OVERWRITE completely\n"
                f"• CANCEL: Pick another sign",
                parent=win
            )
            if resp is None: return
            append_mode[0] = resp

        final_name[0] = name
        close_dialog_cleanly(win)

    def on_cancel():
        close_dialog_cleanly(win)

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Confirm", command=on_confirm, bg="#4CAF50", fg="white",
              font=("Segoe UI", 10, "bold"), padx=12, pady=4).pack(side=tk.LEFT, padx=8)
    tk.Button(btn_frame, text="Cancel / Exit", command=on_cancel, bg="#777", fg="white",
              font=("Segoe UI", 10), padx=10, pady=4).pack(side=tk.LEFT, padx=8)

    win.wait_window()
    return final_name[0], append_mode[0]


# -------------------------------------------------------------
# 3. MediaPipe Setup
# -------------------------------------------------------------
TASK_PATH = "hand_landmarker.task"
if not os.path.exists(TASK_PATH):
    print("⏳ Downloading Google Hand Model (8 MB)...")
    import urllib.request
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", TASK_PATH)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=TASK_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.35,
    min_tracking_confidence=0.35
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)


# -------------------------------------------------------------
# 4. Recording Pipeline (Passes res.handedness!)
# -------------------------------------------------------------
def record_sign_session(camera, sign_name, append_mode=False, instruction_text="Form hand sign & press SPACE"):
    SAMPLES_TO_RECORD = 100
    last_timestamp = 0

    while True:
        redo_requested = False

        # Phase A: Wait for SPACE
        while True:
            success, frame = camera.read()
            if not success: continue
            frame = cv2.flip(frame, 1)

            mode_label = "APPENDING TO" if append_mode else "RECORDING"
            cv2.rectangle(frame, (15, 15), (620, 135), (25, 25, 25), -1)
            cv2.putText(frame, f"{mode_label}: [{sign_name}]", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            cv2.putText(frame, instruction_text, (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, "SPACE: Start  |  'r': Redo  |  'q': Quit", (25, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
            cv2.imshow("ISL Studio", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32: break
            elif key == ord('r'): continue
            elif key == ord('q'): return False

        # Phase B: 5-Second Countdown with Live Hand Counter
        for count in [5, 4, 3, 2, 1]:
            start_t = time.time()
            while time.time() - start_t < 1.0:
                success, frame = camera.read()
                if not success: continue
                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                curr_time = int(time.time() * 1000)
                if curr_time <= last_timestamp: curr_time = last_timestamp + 1
                last_timestamp = curr_time
                res = landmarker.detect_for_video(mp_img, curr_time)

                num_hands = len(res.hand_landmarks) if res.hand_landmarks else 0
                hand_color = (0, 255, 0) if num_hands >= 1 else (0, 0, 255)

                cv2.putText(frame, f"Recording in...", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                cv2.putText(frame, str(count), (290, 260), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), 7)
                cv2.putText(frame, f"Hands detected: {num_hands}", (30, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.75, hand_color, 2)
                cv2.putText(frame, "Gently vary hand tilt/looseness  |  'r': Abort", (30, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
                cv2.imshow("ISL Studio", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    redo_requested = True
                    break
                elif key == ord('q'):
                    return False
            if redo_requested: break

        if redo_requested: continue

        # Phase C: Burst Capture (100 Samples)
        recorded_data = []
        h, w, _ = frame.shape

        while len(recorded_data) < SAMPLES_TO_RECORD:
            success, frame = camera.read()
            if not success: continue
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            curr_time = int(time.time() * 1000)
            if curr_time <= last_timestamp: curr_time = last_timestamp + 1
            last_timestamp = curr_time

            res = landmarker.detect_for_video(mp_img, curr_time)

            if res.hand_landmarks:
                # CRITICAL: Pass res.handedness here!
                feats = extract_features_from_landmarks(res.hand_landmarks, res.handedness)
                if feats:
                    recorded_data.append(feats)
                    for hand in res.hand_landmarks:
                        for lm in hand:
                            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "⚠️ Hand missing! Put hands in view", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.putText(frame, f"RECORDING [{sign_name}]: {len(recorded_data)}/{SAMPLES_TO_RECORD}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            cv2.putText(frame, "Gently flex & tilt hand for variance  |  'r': Redo", (30, 435),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("ISL Studio", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('r'):
                redo_requested = True
                break
            elif key == ord('q'):
                return False

        if redo_requested: continue

        # Phase D: Review Confirmation
        review_start = time.time()
        confirm_save = True
        while time.time() - review_start < 3.5:
            success, frame = camera.read()
            if not success: continue
            frame = cv2.flip(frame, 1)

            cv2.rectangle(frame, (15, 15), (600, 120), (25, 25, 25), -1)
            cv2.putText(frame, f"Captured 100 samples for [{sign_name}]!", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to Save  |  Press 'r' to REDO", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
            cv2.imshow("ISL Studio", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32: break
            elif key == ord('r'):
                redo_requested = True
                confirm_save = False
                break
            elif key == ord('q'):
                return False

        if redo_requested or not confirm_save: continue

        # Save or Append
        file_path = os.path.join(DATASET_DIR, f"{sign_name}.pickle")
        if append_mode and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                existing_data = pickle.load(f)
            recorded_data = existing_data + recorded_data
            print(f"💾 Appended! '{sign_name}' now has {len(recorded_data)} total samples.")
        else:
            print(f"💾 Saved '{sign_name}' ({len(recorded_data)} samples) -> {file_path}")

        with open(file_path, "wb") as f:
            pickle.dump(recorded_data, f)
        return True


# -------------------------------------------------------------
# 5. In-Studio Trainer (Zero Data Leakage + Cushion)
# -------------------------------------------------------------
def train_model_in_studio():
    pickle_files = glob.glob(os.path.join(DATASET_DIR, "*.pickle"))
    if len(pickle_files) < 2:
        safe_show_warning("Cannot Train", f"Found only {len(pickle_files)} signs in '{DATASET_DIR}'. You need at least 2 signs to train!")
        return None, 0.0

    raw_data, raw_labels = [], []
    for filepath in pickle_files:
        sign_name = os.path.basename(filepath).replace(".pickle", "")
        with open(filepath, "rb") as f:
            samples = pickle.load(f)
        for s in samples:
            raw_data.append(np.array(s, dtype=np.float32))
            raw_labels.append(sign_name)

    X_raw = np.asarray(raw_data)
    y_raw = np.asarray(raw_labels)

    # Split FIRST to guarantee zero data leakage
    x_train_raw, x_test, y_train_raw, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, shuffle=True, stratify=y_raw, random_state=42
    )

    x_train, y_train = [], []
    for s_arr, label in zip(x_train_raw, y_train_raw):
        x_train.append(s_arr)
        y_train.append(label)

        # AI Safety Cushion (Natural ±1.5% micro-jitter)
        jitter = s_arr.copy()
        noise = np.random.normal(0, 0.015, size=s_arr.shape).astype(np.float32)
        mask = s_arr != 0.0
        jitter[mask] += noise[mask]
        x_train.append(jitter)
        y_train.append(label)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    print(f"📊 Training on {len(x_train)} augmented samples, testing on {len(x_test)} unseen real samples.")
    model = RandomForestClassifier(n_estimators=120, max_depth=22, n_jobs=-1, random_state=42)
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))
    accuracy_pct = round(accuracy * 100, 2)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model, accuracy_pct


# -------------------------------------------------------------
# 6. Live Test Mode with Majority Voting (Passes res.handedness!)
# -------------------------------------------------------------
def run_live_test(camera, model):
    last_timestamp = 0
    prediction_buffer = deque(maxlen=6)

    print("🚀 Live Test Running! Press 'q' on the video window to stop.")

    while True:
        success, frame = camera.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

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
            # CRITICAL: Pass result.handedness here!
            features = extract_features_from_landmarks(result.hand_landmarks, result.handedness)
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


# =============================================================
# MAIN STUDIO WORKFLOW
# =============================================================
def main():
    chosen_camera_index, selected_mode = ask_startup_popup()
    camera = cv2.VideoCapture(chosen_camera_index)
    if not camera.isOpened():
        print(f"❌ Could not open camera {chosen_camera_index}!")
        return

    # Direct jump to test
    if selected_mode == "test":
        if os.path.exists(MODEL_FILE):
            print(f"📂 Loading existing model '{MODEL_FILE}'...")
            with open(MODEL_FILE, "rb") as f:
                trained_model = pickle.load(f)
            run_live_test(camera, trained_model)
            camera.release()
            cv2.destroyAllWindows()
            print("👋 Live Test closed.")
            return
        else:
            safe_show_info("No Model Found", "No trained model found! Starting Recording Mode.")

    # Recording mode
    while True:
        sign_info = get_standardized_sign_dialog()
        if not sign_info or not sign_info[0]:
            break

        sign_name, append_mode = sign_info
        success = record_sign_session(
            camera, 
            sign_name, 
            append_mode=append_mode,
            instruction_text="Form hand sign & press SPACE to start"
        )
        if not success:
            break

        if not safe_ask_yes_no("Sign Saved!", f"'{sign_name}' saved!\n\nRecord another sign?"):
            break

    # Mandatory Neutral Check
    neutral_path = os.path.join(DATASET_DIR, f"{NEUTRAL_SIGN_NAME}.pickle")
    if not os.path.exists(neutral_path):
        safe_show_info("Essential Step", "Let's record 5 seconds of relaxed hands so the model knows when you are NOT signing.")
        record_sign_session(camera, NEUTRAL_SIGN_NAME, append_mode=False, instruction_text="RELAX hands/fingers casually & press SPACE")

    # Train Model
    if safe_ask_yes_no("Train Model", "All signs and Neutral pose are ready!\n\nTrain model now?"):
        print("🧠 Training model inside Studio (with Auto-Mirroring & Safety Cushion)...")
        trained_model, accuracy = train_model_in_studio()
        if trained_model and safe_ask_yes_no("Model Ready!", f"🎉 Trained with {accuracy}% honest accuracy!\n\nLaunch Live Test now?"):
            cv2.destroyAllWindows()
            run_live_test(camera, trained_model)

    print("👋 Studio closed.")
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()