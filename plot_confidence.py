import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

FILES = { "FaceNet": "results_facenet.csv", "ArcFace": "results_arcface.csv" }

def parse_conf(row):
    """Extracts just the number 99.5 from 'Snehil: 99.5%'"""
    if pd.isna(row): return None
    match = re.search(r"(\d+\.?\d*)%", str(row))
    return float(match.group(1)) if match else None

plt.figure(figsize=(10, 6))

for model, csv in FILES.items():
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        # Filter for successful hits only
        hits = df[(df['success'] == True) & (df['type'] == 'Positive')].copy()
        
        # Extract numbers
        hits['score'] = hits['confidence'].apply(parse_conf)
        
        # Plot Curve
        sns.kdeplot(hits['score'], label=f"{model}", fill=True, alpha=0.3)

plt.title("Model Confidence Distribution (Higher is Better)")
plt.xlabel("Confidence Score (%)")
plt.xlim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("confidence_comparison.png")
print("Saved confidence_comparison.png")