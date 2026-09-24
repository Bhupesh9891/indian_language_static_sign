import numpy as np

TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm
]

def extract_features_from_landmarks(hand_landmarks_list, handedness_list=None):
    """
    112-D Feature Extractor with True Canonical Hand Normalization & Chirality Locking:
    - Single Hand: Left Hands are canonically mirrored to Right Hands (100% Ambidextrous).
    - Two Hands: Locked into Slot 0 (Left) and Slot 1 (Right). Never swaps on cross/touch.
    - Knuckle-based (5 vs 17) rigid orientation check prevents fist/curl glitches.
    """
    if not hand_landmarks_list:
        return None

    # Step 1: Pair landmarks with their detected handedness label
    hands_with_labels = []
    for idx, lms in enumerate(hand_landmarks_list):
        detected_label = None
        if handedness_list and idx < len(handedness_list):
            try:
                # MediaPipe Tasks format
                detected_label = handedness_list[idx][0].category_name
            except (AttributeError, IndexError):
                pass
        hands_with_labels.append((lms, detected_label))

    # Step 2: Assign Hands to Slots (Chirality Locking)
    if len(hands_with_labels) == 1:
        ordered_hands = [hands_with_labels[0]]
        is_single_hand = True
    else:
        is_single_hand = False
        h1, h2 = hands_with_labels[0], hands_with_labels[1]
        # Always enforce: Slot 0 = Left Hand, Slot 1 = Right Hand
        if h1[1] == "Right" and h2[1] == "Left":
            ordered_hands = [h2, h1]
        elif h1[1] == "Left" and h2[1] == "Right":
            ordered_hands = [h1, h2]
        else:
            # Fallback to centroid X sorting if labels are ambiguous
            ordered_hands = sorted(hands_with_labels, key=lambda item: np.mean([lm.x for lm in item[0]]))

    feature_vector = []
    hand_meta = []

    for i in range(2):
        if i < len(ordered_hands):
            lms, hand_label = ordered_hands[i]
            x_vals = np.array([lm.x for lm in lms], dtype=np.float32)
            y_vals = np.array([lm.y for lm in lms], dtype=np.float32)

            min_x, max_x = np.min(x_vals), np.max(x_vals)
            min_y, max_y = np.min(y_vals), np.max(y_vals)

            box_scale = max(max_x - min_x, max_y - min_y, 1e-5)

            # Step 3: Canonical Hand Normalization (Single Hand Only)
            if hand_label is not None:
                is_left_hand = (hand_label == "Left")
            else:
                # RIGID SKELETON FALLBACK:
                # Landmark 5 = Index knuckle, Landmark 17 = Pinky knuckle.
                # Palm knuckles never cross or curl like fingertips do!
                is_left_hand = (x_vals[5] > x_vals[17])

            # If it's a single Left Hand, flip X into canonical Right Hand space:
            if is_single_hand and is_left_hand:
                norm_x = (max_x - x_vals) / box_scale
            else:
                norm_x = (x_vals - min_x) / box_scale

            norm_y = (y_vals - min_y) / box_scale

            hand_meta.append({
                "wrist": np.array([x_vals[0], y_vals[0]]),
                "thumb": np.array([x_vals[4], y_vals[4]]),
                "index": np.array([x_vals[8], y_vals[8]]),
                "scale": box_scale
            })

            # 42 Normalized Coordinates
            for nx, ny in zip(norm_x, norm_y):
                feature_vector.extend([nx, ny])

            coords = np.column_stack([norm_x, norm_y])

            # 4 Thumb-to-Fingertip Distances (Fixes 6, 7, 8, 9)
            thumb_pt = coords[4]
            feature_vector.extend([
                np.linalg.norm(thumb_pt - coords[8]),
                np.linalg.norm(thumb_pt - coords[12]),
                np.linalg.norm(thumb_pt - coords[16]),
                np.linalg.norm(thumb_pt - coords[20])
            ])

            # 1 Fingertip Huddle / Cluster Metric (Fixes 0 & O)
            tip_dists = [
                np.linalg.norm(coords[TIPS[a]] - coords[TIPS[b]])
                for a in range(len(TIPS)) for b in range(a + 1, len(TIPS))
            ]
            feature_vector.append(np.mean(tip_dists) if tip_dists else 0.0)

            # 5 Finger Extension Ratios (Wrist to tips)
            wrist_pt = coords[0]
            for t_idx in TIPS:
                feature_vector.append(np.linalg.norm(coords[t_idx] - wrist_pt))
        else:
            feature_vector.extend([0.0] * 52)

    # 8 Inter-Hand Connection Features
    if len(ordered_hands) >= 2:
        h1, h2 = hand_meta[0], hand_meta[1]
        avg_scale = (h1["scale"] + h2["scale"]) / 2.0

        delta_wrist = (h2["wrist"] - h1["wrist"]) / avg_scale
        wrist_dist = np.linalg.norm(delta_wrist)

        d_idx_idx = np.linalg.norm(h1["index"] - h2["index"]) / avg_scale
        d_th_th = np.linalg.norm(h1["thumb"] - h2["thumb"]) / avg_scale
        d_h2idx_h1th = np.linalg.norm(h2["index"] - h1["thumb"]) / avg_scale
        d_h1idx_h2th = np.linalg.norm(h1["index"] - h2["thumb"]) / avg_scale
        scale_ratio = h2["scale"] / h1["scale"]

        feature_vector.extend([
            float(delta_wrist[0]), float(delta_wrist[1]), float(wrist_dist),
            float(d_idx_idx), float(d_th_th),
            float(d_h2idx_h1th), float(d_h1idx_h2th),
            float(scale_ratio)
        ])
    else:
        feature_vector.extend([0.0] * 8)

    return feature_vector