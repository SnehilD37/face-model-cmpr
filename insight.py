import utils

DATASET_DIR = r"C:\Users\91896\Desktop\FaceModelCompare\known"
PROBE_DIR = r"C:\Users\91896\Desktop\FaceModelCompare\probe"

# Output file name
CSV_FILE = "results_insightface.csv"

# --- EXECUTION ---
if __name__ == "__main__":
    # We don't need to specify 'ArcFace' or 'RetinaFace' anymore 
    # because utils.py is hardcoded to use InsightFace's best tools.
    
    utils.run_pipeline(
        dataset_dir=DATASET_DIR, 
        probe_dir=PROBE_DIR, 
        model_name="InsightFace", 
        csv_filename=CSV_FILE
    )