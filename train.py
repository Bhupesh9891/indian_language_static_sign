import os
import glob
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_DIR = "dataset"
MODEL_FILE = "my_hand_model.p"

pickle_files = glob.glob(os.path.join(DATASET_DIR, "*.pickle"))
if len(pickle_files) < 2:
    print(f"❌ Found only {len(pickle_files)} signs in '{DATASET_DIR}'. You need at least 2 signs to train!")
    exit(1)

raw_data, raw_labels = [], []
print("📂 Loading dataset...")

for filepath in pickle_files:
    sign_name = os.path.basename(filepath).replace(".pickle", "")
    with open(filepath, "rb") as f:
        samples = pickle.load(f)
    for s in samples:
        raw_data.append(np.array(s, dtype=np.float32))
        raw_labels.append(sign_name)
    print(f"  👉 Loaded '{sign_name}': {len(samples)} real samples")

X_raw = np.asarray(raw_data)
y_raw = np.asarray(raw_labels)

# 1. SPLIT FIRST (Zero data leakage)
x_train_raw, x_test, y_train_raw, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, shuffle=True, stratify=y_raw, random_state=42
)

# 2. AUGMENT ONLY TRAINING SAMPLES
x_train, y_train = [], []
for s_arr, label in zip(x_train_raw, y_train_raw):
    x_train.append(s_arr)
    y_train.append(label)

    # Auto-Mirroring for single hands
    if np.all(s_arr[52:104] == 0.0):
        mirrored = s_arr.copy()
        x_coords = mirrored[0:42:2]
        mirrored[0:42:2] = np.max(x_coords) - x_coords + np.min(x_coords)
        x_train.append(mirrored)
        y_train.append(label)

    # AI Safety Cushion (Synthetic Jitter)
    jitter = s_arr.copy()
    noise = np.random.normal(0, 0.015, size=s_arr.shape).astype(np.float32)
    mask = s_arr != 0.0
    jitter[mask] += noise[mask]
    x_train.append(jitter)
    y_train.append(label)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(f"\n📊 Training on {len(x_train)} augmented samples, testing on {len(x_test)} unseen real samples.")
print("🧠 Training Random Forest on CPU...")

model = RandomForestClassifier(n_estimators=120, max_depth=22, n_jobs=-1, random_state=42)
model.fit(x_train, y_train)

score = accuracy_score(y_test, model.predict(x_test))
accuracy_pct = round(score * 100, 2)
print(f"🎉 Model Trained! Honest Test Accuracy: {accuracy_pct}%")

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Saved updated model to '{MODEL_FILE}'!")