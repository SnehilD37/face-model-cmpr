import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np

# List of your results files
FILES = {
    "FaceNet": "results_facenet.csv",
    "ArcFace": "results_arcface.csv",
    "CosFace": "results_cosface.csv",
    "SphereFace": "results_sphereface.csv"
}

def get_clean_label(text):
    """
    Standardizes labels to remove duplicates (e.g. 'snehil' -> 'Snehil').
    """
    # 1. Handle Empty/NaN
    if pd.isna(text) or str(text).lower() in ["nan", "none", "unknown"]:
        return "Negative"
    
    # 2. Handle Filenames (Ground Truth)
    text_str = str(text)
    if "." in text_str: # It's a filename like 'snehil_1.jpg'
        base = os.path.splitext(text_str)[0]
        if "negative" in base.lower() or "empty" in base.lower():
            return "Negative"
        
        # Extract name (remove numbers)
        parts = base.split('_')
        name_parts = [p for p in parts if not p.isdigit() and len(p) > 1]
        if name_parts:
            return name_parts[0].title() # <--- FORCE TITLE CASE
        return "Unknown"
        
    # 3. Handle Predictions (Model Output)
    # If "Snehil, Archisman", take "Snehil"
    first_match = text_str.split(',')[0].strip()
    return first_match.title() # <--- FORCE TITLE CASE

def plot_confusion_matrix(df, model_name):
    """Generates a Cleaned Heatmap"""
    print(f"   [Plotting] Confusion Matrix for {model_name}...")
    
    y_true = []
    y_pred = []

    for index, row in df.iterrows():
        # Get Ground Truth from Filename
        true_label = get_clean_label(row['image'])
        
        # Get Prediction from 'detected' column
        # Map "nan" or "None" to "Negative"
        detected_val = row['detected']
        if pd.isna(detected_val) or str(detected_val) == "None":
            pred_label = "Negative"
        else:
            pred_label = get_clean_label(detected_val)

        y_true.append(true_label)
        y_pred.append(pred_label)

    # Sort labels alphabetically
    labels = sorted(list(set(y_true + y_pred)))
    
    # Move 'Negative' to the end for better visualization if present
    if "Negative" in labels:
        labels.remove("Negative")
        labels.append("Negative")

    # Generate Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Name')
    plt.ylabel('True Name (File)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

def plot_time_distribution(df, model_name):
    """Plots the Speed (Time) Distribution Graph"""
    print(f"   [Plotting] Time Distribution for {model_name}...")
    
    plt.figure(figsize=(8, 5))
    
    # Plot Histogram of 'time' column
    sns.histplot(df['time'], kde=True, color='purple', bins=15)
    
    # Add Average Line
    avg_time = df['time'].mean()
    plt.axvline(avg_time, color='red', linestyle='dashed', linewidth=2)
    plt.text(avg_time * 1.05, 1, f'Avg: {avg_time:.2f}s', color='red')
    
    plt.title(f'Inference Speed - {model_name}')
    plt.xlabel('Time per Image (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'time_dist_{model_name}.png')
    plt.close()

if __name__ == "__main__":
    for model, csv_file in FILES.items():
        if os.path.exists(csv_file):
            print(f"--- Processing {model} ---")
            try:
                df = pd.read_csv(csv_file)
                
                # 1. Confusion Matrix (Accuracy)
                plot_confusion_matrix(df, model)
                
                # 2. Time Distribution (Speed)
                plot_time_distribution(df, model)
                
            except Exception as e:
                print(f"Error processing {model}: {e}")
        else:
            print(f"Skipping {model} (File not found)")