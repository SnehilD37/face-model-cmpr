import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Add this line!
import time
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
THRESHOLDS = {
    "FaceNet": 0.40,
    "ArcFace": 0.65,
    "CosFace": 0.40,
    "SphereFace": 0.50
}

# --- REPLACE THIS SECTION IN UTILS.PY ---

# strict mapping to valid deepface library strings
MODEL_MAP = {
    "FaceNet": "Facenet",      # Changed "FaceNet" -> "Facenet" (Lowercase 'n')
    "DeepFace": "DeepFace",    # This is correct
    "ArcFace": "ArcFace",      # This is correct
    "VGG-Face": "VGG-Face",    # This is correct
    "SphereFace": "SFace",     # Maps SphereFace to SFace
    "CosFace": "ArcFace"       # Uses ArcFace as proxy
}
def load_database(dataset_dir, model_name):
    """ Loads known faces into memory. """
    print(f"[{model_name}] Building Database...")
    database = {} 
    
    if not os.path.exists(dataset_dir):
        return {}

    lib_model = MODEL_MAP.get(model_name, model_name)

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path): continue
        
        database[person_name] = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            try:
                results = DeepFace.represent(
                    img_path=img_path,
                    model_name=lib_model,
                    detector_backend='retinaface',
                    enforce_detection=False
                )
                if results:
                    database[person_name].append(results[0]['embedding'])
            except:
                continue
    return database

def run_pipeline(dataset_dir, probe_dir, model_name, csv_filename):
    db = load_database(dataset_dir, model_name)
    lib_model = MODEL_MAP.get(model_name, model_name)
    
    output_vis_dir = f"output_visuals_{model_name}"
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print(f"[{model_name}] Testing (Visuals -> {output_vis_dir})...")
    
    report_data = []
    
    for img_file in os.listdir(probe_dir):
        img_path = os.path.join(probe_dir, img_file)
        full_img = cv2.imread(img_path)
        if full_img is None: continue
        
        # --- 1. DETECT & EMBED ---
        # --- 1. DETECT & EMBED ---
        start_time = time.time()
        try:
            face_objs = DeepFace.represent(
                img_path=img_path,
                model_name=lib_model,
                detector_backend='retinaface',
                enforce_detection=False
            )
        except Exception as e:
            # THIS PRINT STATEMENT IS CRITICAL FOR DEBUGGING
            print(f"\n[CRITICAL ERROR] {model_name} crashed on {img_file}: {e}")
            face_objs = []
        inference_time = time.time() - start_time
        
        # --- 2. IDENTIFY FACES ---
        detected_names = []
        
        for face_obj in face_objs:
            probe_emb = face_obj['embedding']
            
            # Find best match in DB
            best_match = "Unknown"
            min_dist = 100
            
            for name, db_embs in db.items():
                for db_emb in db_embs:
                    sim = cosine_similarity([probe_emb], [db_emb])[0][0]
                    dist = 1 - sim
                    if dist < min_dist:
                        min_dist = dist
                        if dist < THRESHOLDS.get(model_name, 0.4):
                            best_match = name
            
            if best_match != "Unknown":
                detected_names.append(best_match)
            
            # Draw Box
            r = face_obj['facial_area']
            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
            cv2.rectangle(full_img, (r['x'], r['y']), (r['x']+r['w'], r['y']+r['h']), color, 2)
            cv2.putText(full_img, best_match, (r['x'], r['y']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # --- 3. CHECK SUCCESS (UPDATED LOGIC) ---
        is_negative_image = "negative" in img_file.lower() or "empty" in img_file.lower()
        success = False
        
        if is_negative_image:
            # SUCCESS = Found NO known names (Empty list or all Unknown)
            if len(detected_names) == 0:
                success = True
            else:
                success = False # FAIL: Found a person in an empty image!
        else:
            # SUCCESS = Found the person mentioned in filename
            for name in detected_names:
                if name.lower() in img_file.lower():
                    success = True
                    break

        # Save Visual
        cv2.imwrite(os.path.join(output_vis_dir, img_file), full_img)
        
        report_data.append({
            "image": img_file,
            "type": "Negative" if is_negative_image else "Positive",
            "detected": ", ".join(detected_names) if detected_names else "None",
            "success": success,
            "time": round(inference_time, 4)
        })

    # Save CSV
    df = pd.DataFrame(report_data)
    df.to_csv(csv_filename, index=False)
    print(f"[{model_name}] Saved results to {csv_filename}")