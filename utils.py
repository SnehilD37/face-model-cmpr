import os
import time
import cv2
import pandas as pd
import numpy as np
import re
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import insightface
from insightface.app import FaceAnalysis
import math

# --- 1. TUNING KNOBS ---
# InsightFace Settings
INSIGHT_DET_CONF = 0.60 
INSIGHT_REC_THRESH = 0.55

# DeepFace Settings (FaceNet, CosFace, etc.)
DEEPFACE_DET_CONF = 0.95 
MIN_FACE_AREA_RATIO = 0.002

# --- 2. INITIALIZE MODELS ---
print("Initializing Engines...")

# Engine A: InsightFace (Fast)
app_insight = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))

# Engine B: MTCNN (for DeepFace models)
detector_mtcnn = MTCNN()

# Thresholds (Lower = Stricter for DeepFace models)
THRESHOLDS = {
    "FaceNet": 0.40,
    "ArcFace": 0.60,
    "CosFace": 0.40,
    "SphereFace": 0.40,
    "InsightFace": INSIGHT_DET_CONF 
}

MODEL_MAP = {
    "FaceNet": "Facenet",
    "ArcFace": "ArcFace",
    "SphereFace": "SFace",
    "CosFace": "ArcFace" 
}

# --- 3. HELPER FUNCTIONS ---

def get_expected_names(filename):
    base = os.path.splitext(filename)[0]
    base = re.sub(r'_\d+$', '', base)
    return base.lower()

def alignment_procedure(img, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = math.atan2(delta_y, delta_x)
    angle_degree = (angle * 180) / math.pi
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

def load_database(dataset_dir, model_name):
    print(f"[{model_name}] Building Database...")
    database = {}
    if not os.path.exists(dataset_dir): return {}
    
    use_insight = (model_name == "InsightFace")
    
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path): continue
        
        database[person_name] = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            emb = None
            if use_insight:
                faces = app_insight.get(img)
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
                    emb = faces[0].embedding
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = detector_mtcnn.detect_faces(img_rgb)
                if results:
                    face = max(results, key=lambda x: x['box'][2] * x['box'][3])
                    keypoints = face['keypoints']
                    img_aligned = alignment_procedure(img_rgb, keypoints['left_eye'], keypoints['right_eye'])
                    res_aligned = detector_mtcnn.detect_faces(img_aligned)
                    if res_aligned:
                         face_final = max(res_aligned, key=lambda x: x['box'][2] * x['box'][3])
                         x, y, w, h = face_final['box']
                         x, y = max(0, x), max(0, y)
                         face_crop = img_aligned[y:y+h, x:x+w]
                         try:
                             deep_model = MODEL_MAP.get(model_name, model_name)
                             res_deep = DeepFace.represent(img_path=face_crop, model_name=deep_model, detector_backend='skip', enforce_detection=False)
                             if res_deep: emb = res_deep[0]['embedding']
                         except: pass

            if emb is not None:
                database[person_name].append(emb)
    return database

# --- 4. MAIN PIPELINE ---

def run_pipeline(dataset_dir, probe_dir, model_name, csv_filename):
    db = load_database(dataset_dir, model_name)
    use_insight = (model_name == "InsightFace")
    
    output_vis_dir = f"output_visuals_{model_name}"
    os.makedirs(output_vis_dir, exist_ok=True)
    
    all_files = [f for f in os.listdir(probe_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[{model_name}] Processing {len(all_files)} images...")
    
    report_data = []
    
    for index, img_file in enumerate(all_files):
        print(f"Processing {index + 1}/{len(all_files)}: {img_file} ...", end=" ")
        try:
            img_path = os.path.join(probe_dir, img_file)
            full_img = cv2.imread(img_path)
            if full_img is None: continue
            
            start_time = time.time()
            final_detections = []
            
            # --- 1. GET EMBEDDINGS ---
            face_objs = [] # List of {emb, box}
            
            if use_insight:
                faces = app_insight.get(full_img)
                for face in faces:
                    if face.det_score > INSIGHT_DET_CONF:
                        face_objs.append({'emb': face.embedding, 'box': face.bbox.astype(int)})
            else:
                img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                mtcnn_results = detector_mtcnn.detect_faces(img_rgb)
                for res in mtcnn_results:
                    x, y, w, h = res['box']
                    if res['confidence'] > DEEPFACE_DET_CONF and (w*h) > (full_img.shape[0]*full_img.shape[1]*MIN_FACE_AREA_RATIO):
                         x, y = max(0, x), max(0, y)
                         face_crop = img_rgb[y:y+h, x:x+w]
                         if face_crop.size > 0:
                             try:
                                 deep_model = MODEL_MAP.get(model_name, model_name)
                                 emb_objs = DeepFace.represent(img_path=face_crop, model_name=deep_model, detector_backend='skip', enforce_detection=False)
                                 if emb_objs:
                                     face_objs.append({'emb': emb_objs[0]['embedding'], 'box': [x, y, x+w, y+h]})
                             except: pass

            inference_time = time.time() - start_time
            
            # --- 2. MATCHING LOGIC (With Raw Logging) ---
            thresh = INSIGHT_REC_THRESH if use_insight else THRESHOLDS.get(model_name, 0.40)
            
            for obj in face_objs:
                probe_emb = obj['emb']
                
                # Capture the absolute best match BEFORE applying threshold
                raw_best_name = "None"
                raw_best_dist = 100
                
                for name, db_embs in db.items():
                    for db_emb in db_embs:
                        sim = cosine_similarity([probe_emb], [db_emb])[0][0]
                        dist = 1 - sim
                        
                        if dist < raw_best_dist:
                            raw_best_dist = dist
                            raw_best_name = name
                
                # Convert distance to % score
                raw_score = max(0, (1 - raw_best_dist) * 100)
                
                # Apply Threshold for final decision
                if raw_best_dist < thresh:
                    final_name = raw_best_name
                else:
                    final_name = "Unknown"
                    
                final_detections.append({
                    'name': final_name,
                    'score': raw_score, # This is the score of the match (even if rejected)
                    'box': obj['box'],
                    'raw_match_name': raw_best_name, # FOR ROC: Who did it look like?
                    'raw_match_score': raw_score     # FOR ROC: How close was it?
                })

            # --- 3. REPORTING ---
            final_detections.sort(key=lambda x: x['score'], reverse=True)
            expected_str = get_expected_names(img_file)
            
            found_people = []
            avg_confs = []
            raw_data_list = [] # String for CSV
            
            for det in final_detections:
                name = det['name']
                score = int(det['score'])
                box = det['box']
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                # Log raw data for ROC: "Name:Score%"
                raw_data_list.append(f"{det['raw_match_name']}:{det['raw_match_score']:.1f}%")

                if name == "Unknown":
                    color = (128, 128, 128)
                elif name.lower() in expected_str:
                    color = (0, 255, 0)
                    found_people.append(name)
                    avg_confs.append(score)
                else:
                    color = (0, 0, 255)
                
                cv2.rectangle(full_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(full_img, f"{name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imwrite(os.path.join(output_vis_dir, img_file), full_img)
            
            avg_conf_val = sum(avg_confs)/len(avg_confs) if avg_confs else 0
            
            report_data.append({
                "image": img_file,
                "expected_people": expected_str,
                "found_people": ", ".join(found_people),
                "avg_confidence": f"{avg_conf_val:.1f}%" if avg_conf_val > 0 else "",
                "raw_matches": ", ".join(raw_data_list), # NEW COLUMN FOR ROC
                "total_faces_kept": len(final_detections),
                "time": round(inference_time, 4)
            })
            
            print(f"Done ({len(final_detections)} faces)")
            pd.DataFrame(report_data).to_csv(csv_filename, index=False)

        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"[{model_name}] Saved to {csv_filename}")