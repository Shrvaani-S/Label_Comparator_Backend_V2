import cv2
import numpy as np
import os
import math
import tempfile
from pathlib import Path

# =====================================================================
# OLD YOLO-BASED DETECTION ENGINE (COMMENTED OUT)
# Replaced by SIFT + Template Matching engine from backend_main.
# =====================================================================
# from ultralytics import YOLO
# import gc
#
# # -----------------------------
# # Run Detection (For Streamlit/PIL)
# # -----------------------------
# def run_detection_pil_yolo(pil_image, label_name="Document"):
#     """Handles PIL images using a Load-and-Dump memory approach with detailed logging"""
#     print(f"\n{'='*60}")
#     print(f"🤖 YOLO AI SYMBOL DETECTION STARTING")
#     print(f"{'='*60}")
#
#     # 1. Load models into RAM
#     print("[1/3] Loading neural networks into memory...")
#     try:
#         model16 = YOLO("16sym_models/best.pt")
#         model4 = YOLO("4sym_models/best.pt")
#     except Exception as e:
#         print(f"❌ ERROR LOADING MODELS: {e}")
#         return []
#
#     # 2. Process the image
#     print(f"[2/3] Scanning image pixels...")
#     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
#         pil_image.save(tmp.name)
#         # verbose=False hides the messy default Ultralytics logs so our clean logs shine
#         results16 = model16(tmp.name, conf=0.3, verbose=False)[0]
#         results4 = model4(tmp.name, conf=0.3, verbose=False)[0]
#
#     detections = []
#     print("\n--- EXTRACTED SYMBOLS ---")
#
#     for box in results16.boxes:
#         cls = model16.names[int(box.cls)]
#         bbox = box.xyxy[0].tolist()
#         conf = float(box.conf)
#         print(f"  ✓ {cls:<20} | Conf: {conf:.2f} | Coords: {[int(x) for x in bbox]}")
#         detections.append({"class": cls, "bbox": bbox, "label": "Symbol"})
#
#     for box in results4.boxes:
#         cls = model4.names[int(box.cls)]
#         bbox = box.xyxy[0].tolist()
#         conf = float(box.conf)
#         print(f"  ✓ {cls:<20} | Conf: {conf:.2f} | Coords: {[int(x) for x in bbox]}")
#         detections.append({"class": cls, "bbox": bbox, "label": "Symbol"})
#
#     if not detections:
#         print("  [!] No symbols detected in this document.")
#     else:
#         print(f"  Total Symbols Found: {len(detections)}")
#
#     # 3. CRITICAL: Delete models and force RAM cleanup
#     print("\n[3/3] Flushing models from RAM...")
#     del model16
#     del model4
#     gc.collect()
#
#     print(f"{'='*60}\n")
#     return detections
#
# # OLD compare_labels — simple class-name matching, 40px threshold, returned misplaced
# def compare_labels_yolo(base_det, edited_det, threshold=40):
#     added = []
#     removed = []
#     misplaced = []
#     print(f"\n🔍 RUNNING SYMBOL DISCREPANCY ENGINE")
#     print("-" * 40)
#     base_classes = [d["class"] for d in base_det]
#     edited_classes = [d["class"] for d in edited_det]
#     for d in edited_det:
#         if d["class"] not in base_classes:
#             d = d.copy()
#             d["label"] = "Added"
#             added.append(d)
#             print(f"  [+] ADDED:     {d['class']}")
#     for d in base_det:
#         if d["class"] not in edited_classes:
#             d = d.copy()
#             d["label"] = "Removed"
#             removed.append(d)
#             print(f"  [-] DELETED:   {d['class']}")
#     for b in base_det:
#         for e in edited_det:
#             if b["class"] == e["class"]:
#                 c1 = get_center(b["bbox"])
#                 c2 = get_center(e["bbox"])
#                 dist = math.dist(c1, c2)
#                 if dist > threshold:
#                     e = e.copy()
#                     e["label"] = "Repositioned"
#                     misplaced.append(e)
#                     print(f"  [~] MISPLACED: {e['class']} (Shifted {int(dist)} pixels)")
#     print("-" * 40)
#     print(f"  Summary: {len(added)} Added | {len(removed)} Deleted | {len(misplaced)} Misplaced\n")
#     return added, removed, misplaced
# =====================================================================


# =====================================================================
# MODULE-LEVEL: Load symbol templates once at import time
# =====================================================================
_SYMBOL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbols")


def _load_symbol_files(folder):
    symbol_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']:
        symbol_files.extend(Path(folder).glob(ext))
    unique = {}
    for sf in symbol_files:
        if sf.stem not in unique:
            unique[sf.stem] = sf
    return sorted(unique.values())


_SYMBOL_FILES = _load_symbol_files(_SYMBOL_FOLDER) if os.path.isdir(_SYMBOL_FOLDER) else []


# =====================================================================
# SIFT + TEMPLATE MATCHING DETECTION ENGINE
# Ported from backend_main/verify_symbol_changes.py
# =====================================================================

def run_robust_detection(image_path, symbol_files):
    """
    SIFT+FLANN primary detection with Template Matching fallback.
    Returns list of (symbol_id, info_dict) tuples sorted by confidence.
    """
    label_img = cv2.imread(image_path)
    if label_img is None:
        return []

    label_gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
    label_h, label_w = label_gray.shape

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    label_enhanced = clahe.apply(label_gray)

    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=20)
    kp_label, des_label = sift.detectAndCompute(label_enhanced, None)

    if des_label is None:
        return []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    detected_symbols = {}

    # --- Stage 1: SIFT Feature Matching ---
    for symbol_file in symbol_files:
        symbol_path = str(symbol_file)
        symbol_img = cv2.imread(symbol_path, 0)
        if symbol_img is None:
            continue

        symbol_name = symbol_file.stem
        symbol_enhanced = clahe.apply(symbol_img)

        min_dim = min(symbol_enhanced.shape[0], symbol_enhanced.shape[1])
        if min_dim < 150:
            scale = 150 / float(min_dim)
            symbol_enhanced = cv2.resize(symbol_enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        symbol_h, symbol_w = symbol_enhanced.shape
        kp_symbol, des_symbol = sift.detectAndCompute(symbol_enhanced, None)

        if des_symbol is None or len(kp_symbol) < 5:
            continue

        matches = flann.knnMatch(des_symbol, des_label, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        MIN_MATCH_COUNT = 10
        if symbol_w < 50 or symbol_h < 50:
            MIN_MATCH_COUNT = 5

        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_symbol[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_label[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                pts = np.float32([[0, 0], [0, symbol_h - 1], [symbol_w - 1, symbol_h - 1], [symbol_w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                x_coords = [p[0][0] for p in dst]
                y_coords = [p[0][1] for p in dst]

                x1 = int(max(0, min(x_coords)))
                y1 = int(max(0, min(y_coords)))
                x2 = int(min(label_w, max(x_coords)))
                y2 = int(min(label_h, max(y_coords)))

                final_w = x2 - x1
                final_h = y2 - y1

                is_convex = cv2.isContourConvex(np.int32(dst))
                contour_area = cv2.contourArea(np.int32(dst))
                box_area = final_w * final_h

                valid_area = is_convex and box_area > 0 and (contour_area / box_area) > 0.4

                if valid_area and final_w > 10 and final_h > 10 and final_w < label_w * 0.5 and final_h < label_h * 0.5:
                    confidence = len(good_matches) / len(kp_symbol)
                    if len(good_matches) >= 20:
                        confidence = min(1.0, confidence + 0.3)

                    # Raise bar for Sterile variants to avoid cross-symbol matching
                    sift_floor = 0.70 if symbol_name.startswith("Sterile") else 0.25
                    if confidence < sift_floor:
                        continue

                    detected_symbols[symbol_name] = {
                        'name': symbol_name,
                        'confidence': confidence,
                        'coords': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'matches': len(good_matches),
                        'file': symbol_file.name,
                        'outline': dst
                    }

    # --- Stage 2: Template Matching Fallback ---
    tm_list = [
        "Manufacture", "Manufacturer", "LOT", "REF", "EXCM", "Use_By_Date",
        "Logo", "Depuy_LOGO", "MADE_IN", "MR", "Package_Is_Damaged",
        "Consult_IFU", "Date_Of_Manufacture", "Do_Not_Resterilize",
        "Sterile_R", "Sterile_EO", "Sterile_A", "MD", "Med_Dev",
        "Medical_Device", "UDI", "Flammable"
    ]

    for cand_name in tm_list:
        cand_file = next((sf for sf in symbol_files if sf.stem == cand_name), None)
        if not cand_file:
            continue

        cand_img = cv2.imread(str(cand_file), 0)
        if cand_img is None:
            continue

        # Wider scale range for logos
        scales = np.linspace(0.2, 2.5, 40) if "logo" in cand_name.lower() else np.linspace(0.5, 2.5, 30)

        for scale in scales:
            resized_w = int(cand_img.shape[1] * scale)
            resized_h = int(cand_img.shape[0] * scale)
            if resized_h > label_h or resized_w > label_w or resized_h < 10 or resized_w < 10:
                continue

            resized = cv2.resize(cand_img, (resized_w, resized_h))
            res = cv2.matchTemplate(label_gray, resized, cv2.TM_CCOEFF_NORMED)

            # Per-symbol thresholds tuned for stability
            threshold = 0.85
            if cand_name == "Manufacture":         threshold = 0.65
            if cand_name == "Manufacturer":        threshold = 0.65
            if cand_name == "LOT":                 threshold = 0.52
            if cand_name == "REF":                 threshold = 0.53
            if cand_name == "EXCM":                threshold = 0.60
            if cand_name == "Use_By_Date":         threshold = 0.65
            if cand_name == "Logo":                threshold = 0.75
            if cand_name == "Depuy_LOGO":          threshold = 0.75
            if cand_name == "MADE_IN":             threshold = 0.80
            if cand_name == "MR":                  threshold = 0.75
            if cand_name == "Sterile_R":           threshold = 0.75
            if cand_name == "Sterile_EO":          threshold = 0.75
            if cand_name == "Sterile_A":           threshold = 0.75
            if cand_name == "MD":                  threshold = 0.75
            if cand_name == "Med_Dev":             threshold = 0.75
            if cand_name == "Medical_Device":      threshold = 0.75
            if cand_name == "UDI":                 threshold = 0.75
            if cand_name == "Flammable":           threshold = 0.75
            if cand_name == "Package_Is_Damaged":  threshold = 0.75
            if cand_name == "Consult_IFU":         threshold = 0.75
            if cand_name == "Date_Of_Manufacture": threshold = 0.75
            if cand_name == "Do_Not_Resterilize":  threshold = 0.75

            locs = np.where(res >= threshold)
            for pt in zip(*locs[::-1]):
                x1, y1 = int(pt[0]), int(pt[1])
                x2, y2 = x1 + resized_w, y1 + resized_h

                patch = label_gray[y1:y2, x1:x2]
                ph, pw = patch.shape
                ch, cw = ph // 4, pw // 4
                center_patch = patch[ch:ph - ch, cw:pw - cw]

                if np.std(center_patch) < 25:
                    continue

                conf = float(res[y1, x1])
                uid = f"{cand_name}_TM_{x1}_{y1}"

                detected_symbols[uid] = {
                    'name': cand_name,
                    'confidence': conf,
                    'coords': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'matches': int(conf * 100),
                    'file': cand_file.name,
                    'outline': np.float32([[[x1, y1]], [[x1, y2]], [[x2, y2]], [[x2, y1]]])
                }

    # --- Post-processing: Remove LOT false positives inside REF boxes ---
    ref_boxes = [info['coords'] for uid, info in detected_symbols.items() if 'REF' in uid]
    keys_to_delete = []
    for uid, info in detected_symbols.items():
        if 'LOT' in uid:
            cx, cy = info['center']
            for rx1, ry1, rx2, ry2 in ref_boxes:
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    keys_to_delete.append(uid)
                    break
    for uid in keys_to_delete:
        del detected_symbols[uid]

    sorted_symbols = sorted(detected_symbols.items(), key=lambda x: -x[1]['confidence'])

    # --- NMS: Remove overlapping boxes (IoU > 0.3) ---
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

    filtered_symbols = []
    for symbol_id, info in sorted_symbols:
        keep = True
        for _, kept_info in filtered_symbols:
            if compute_iou(info['coords'], kept_info['coords']) > 0.3:
                keep = False
                break
        if keep:
            filtered_symbols.append((symbol_id, info))

    return filtered_symbols


def analyze_symbol_changes(base_detections, child_detections):
    """
    Analyze which symbols are present, removed, repositioned or added.
    Ported from backend_main/verify_symbol_changes.py.

    Accepts [(symbol_id, info_dict)] from run_robust_detection / run_detection_raw.
    Returns (matches, added, removed, repositioned).
    """
    base_grouped = {}
    for uid, info in base_detections:
        name = info['name'].split('_TM_')[0]
        if name not in base_grouped:
            base_grouped[name] = []
        base_grouped[name].append(info)

    child_grouped = {}
    for uid, info in child_detections:
        name = info['name'].split('_TM_')[0]
        if name not in child_grouped:
            child_grouped[name] = []
        child_grouped[name].append(info)

    matches = []
    added = []
    removed = []
    repositioned = []

    POS_TOLERANCE = 45  # pixels

    for name, base_instances in base_grouped.items():
        child_instances = child_grouped.get(name, [])

        for b_info in base_instances:
            b_cx, b_cy = b_info['center']

            closest_c_info = None
            min_dist = float('inf')

            for c_info in child_instances:
                c_cx, c_cy = c_info['center']
                dist = math.sqrt((b_cx - c_cx) ** 2 + (b_cy - c_cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_c_info = c_info

            if closest_c_info is None:
                removed.append({'name': name, 'base_info': b_info})
            else:
                if min_dist > POS_TOLERANCE:
                    repositioned.append({
                        'name': name,
                        'base_info': b_info,
                        'child_info': closest_c_info,
                        'distance': min_dist
                    })
                    child_instances.remove(closest_c_info)
                else:
                    matches.append({
                        'name': name,
                        'base_info': b_info,
                        'child_info': closest_c_info
                    })
                    child_instances.remove(closest_c_info)

        # Any leftover child instances for this name are newly added
        for c_info in child_instances:
            added.append({'name': name, 'child_info': c_info})

    # Child symbols whose name did not appear in base at all
    for name, child_instances in child_grouped.items():
        if name not in base_grouped:
            for c_info in child_instances:
                added.append({'name': name, 'child_info': c_info})

    return matches, added, removed, repositioned


# =====================================================================
# PUBLIC API FUNCTIONS
# =====================================================================

def run_detection_raw(pil_image):
    """
    Accepts a PIL image.
    Returns native [(symbol_id, info_dict)] tuples for direct use with analyze_symbol_changes.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path)
    try:
        return run_robust_detection(tmp_path, _SYMBOL_FILES)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_detection_pil(pil_image, label_name="Document"):
    """
    Backward-compatible adapter.
    Accepts a PIL image.
    Returns [{"class": str, "bbox": [x1,y1,x2,y2], "label": "Symbol"}].
    """
    print(f"\n{'='*60}")
    print(f"SIFT+TM SYMBOL DETECTION STARTING")
    print(f"{'='*60}")

    if not _SYMBOL_FILES:
        print("[!] No symbol files found in symbols/ folder.")
        return []

    raw_detections = run_detection_raw(pil_image)

    detections = []
    print("\n--- EXTRACTED SYMBOLS ---")
    for symbol_id, info in raw_detections:
        name = info["name"]
        x1, y1, x2, y2 = info["coords"]
        conf = info["confidence"]
        print(f"  {name:<35} | Conf: {conf:.2f} | Coords: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
        detections.append({
            "class": name,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "label": "Symbol"
        })

    if not detections:
        print("  [!] No symbols detected in this document.")
    else:
        print(f"  Total Symbols Found: {len(detections)}")

    print(f"{'='*60}\n")
    return detections


# -----------------------------
# Utility / Math Logic
# -----------------------------
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compare_labels(base_det, comp_det, threshold=45):
    """
    Updated wrapper around analyze_symbol_changes.
    Previously used YOLO class-name matching with 40px threshold, returned (added, removed, misplaced).
    Now delegates to analyze_symbol_changes with 45px position tolerance.

    Accepts [{"class","bbox","label"}] dicts (output of run_detection_pil).
    Returns (added, removed, repositioned) as lists of {"class","bbox","label"} dicts.
    """
    def _to_raw_format(detections):
        result = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            name = d["class"]
            uid = f"{name}_{int(x1)}_{int(y1)}"
            info = {
                'name': name,
                'confidence': 1.0,
                'coords': (float(x1), float(y1), float(x2), float(y2)),
                'center': (cx, cy),
                'matches': 0,
                'file': '',
                'outline': np.float32([[[x1, y1]], [[x1, y2]], [[x2, y2]], [[x2, y1]]])
            }
            result.append((uid, info))
        return result

    base_raw = _to_raw_format(base_det)
    comp_raw = _to_raw_format(comp_det)

    _, added_raw, removed_raw, repositioned_raw = analyze_symbol_changes(base_raw, comp_raw)

    added = [{"class": a["name"], "bbox": list(a["child_info"]["coords"]), "label": "Added"} for a in added_raw]
    removed = [{"class": r["name"], "bbox": list(r["base_info"]["coords"]), "label": "Removed"} for r in removed_raw]
    repositioned = [{"class": m["name"], "bbox": list(m["child_info"]["coords"]), "label": "Repositioned"} for m in repositioned_raw]

    print(f"\nSYMBOL COMPARISON: {len(added)} Added | {len(removed)} Removed | {len(repositioned)} Repositioned\n")
    return added, removed, repositioned
