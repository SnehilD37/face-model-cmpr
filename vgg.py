import utils

# Configuration
DATASET_DIR = r"C:\Users\91896\Desktop\FaceModelCompare\known"
PROBE_DIR = r"C:\Users\91896\Desktop\FaceModelCompare\probe"
MODEL_NAME = "VGG-Face"
CSV_OUTPUT = "results_vggface.csv"

if __name__ == "__main__":
    utils.run_pipeline(DATASET_DIR, PROBE_DIR, MODEL_NAME, CSV_OUTPUT)