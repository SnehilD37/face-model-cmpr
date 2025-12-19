import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import roc_curve, auc

# Files to look for
FILES = {
    "FaceNet": "results_facenet.csv",
    "ArcFace": "results_arcface.csv",
    "InsightFace": "results_insightface.csv",
    "CosFace": "results_cosface.csv",
    "SphereFace": "results_sphereface.csv"
}

def clean_confidence(conf_val):
    if pd.isna(conf_val) or conf_val == "": return 0.0
    s = str(conf_val).replace('%', '').strip()
    try: return float(s)
    except: return 0.0

def parse_raw_matches(raw_str):
    """
    Parses "Snehil:99.5%, Subhayan:45.0%" into [('Snehil', 99.5), ('Subhayan', 45.0)]
    """
    if pd.isna(raw_str) or raw_str == "": return []
    items = str(raw_str).split(',')
    parsed = []
    for item in items:
        if ':' in item:
            name, score_str = item.split(':')
            score = clean_confidence(score_str)
            parsed.append((name.strip(), score))
    return parsed

def get_roc_data(df):
    """
    Extracts binary labels (1=Match, 0=Non-Match) and scores for ROC calculation.
    """
    y_true = []
    y_scores = []
    
    for _, row in df.iterrows():
        expected = str(row['expected_people']).lower().strip()
        raw_matches = parse_raw_matches(row.get('raw_matches', ''))
        
        # If no raw matches were logged (e.g. no face detected at all), 
        # we treat it as a missed positive (if expected person exists) or correct negative.
        if not raw_matches:
            # We can't really plot a score if there is no detection, skipping complexity for now
            continue

        for (pred_name, score) in raw_matches:
            # Logic: 
            # If the predicted raw name MATCHES expected -> It is a Positive Sample (1)
            # If the predicted raw name DOES NOT MATCH -> It is a Negative Sample (0)
            
            # Note: "negative" or "empty" expected_people string means NO match is correct.
            is_empty_img = 'negative' in expected or 'empty' in expected
            
            if is_empty_img:
                label = 0 # It should be a non-match
            elif pred_name.lower() in expected:
                label = 1 # Correct Match
            else:
                label = 0 # Impostor / Ghost
            
            y_true.append(label)
            y_scores.append(score / 100.0) # Normalize to 0-1
            
    return y_true, y_scores

def plot_roc_curve(roc_data):
    plt.figure(figsize=(10, 8))
    
    for model, (y_true, y_scores) in roc_data.items():
        if len(y_true) < 2 or sum(y_true) == 0:
            print(f"Skipping ROC for {model} (Insufficient data)")
            continue
            
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Ghosts/Impostors)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig("plot_roc_curve.png")
    print("Saved plot_roc_curve.png")
    plt.close()

# --- REUSED PLOTTING FUNCTIONS ---
def calculate_confusion_matrix(df):
    TP, FP, TN, FN = 0, 0, 0, 0
    for _, row in df.iterrows():
        expected = str(row['expected_people']).lower().strip()
        found = str(row['found_people']).lower().strip()
        is_neg = 'negative' in str(row['image']).lower() or 'empty' in str(row['image']).lower()
        found_is_empty = (found == "nan" or found == "")
        
        if is_neg:
            if found_is_empty: TN += 1
            else: FP += 1
        else:
            if found_is_empty: FN += 1
            elif expected in found: TP += 1
            else: FP += 1
    return np.array([[TP, FP], [FN, TN]])

def plot_bar_comparison(data):
    if not data: return
    df = pd.DataFrame(data)
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df_melt[df_melt['Metric'].isin(['Recall', 'Avg Conf'])], x='Model', y='Value', hue='Metric', palette="viridis")
    plt.title("Accuracy & Confidence")
    plt.ylim(0, 110); plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df_melt[df_melt['Metric'] == 'Time'], x='Model', y='Value', color='salmon')
    plt.title("Processing Speed (Lower is Better)")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig("plot_comparison_bars.png")
    plt.close()

def plot_efficiency_frontier(data):
    if not data: return
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Time', y='Recall', hue='Model', s=200, style='Model', palette="deep")
    for i in range(df.shape[0]):
        plt.text(df.Time[i], df.Recall[i]+1.5, df.Model[i], fontsize=9, ha='center', fontweight='bold')
    plt.title("Efficiency Frontier")
    plt.xlabel("Time (s)"); plt.ylabel("Recall (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("plot_efficiency.png")
    plt.close()

def plot_confusion_matrices(matrices):
    if not matrices: return
    num = len(matrices)
    fig, axes = plt.subplots(1, num, figsize=(4 * num, 4))
    if num == 1: axes = [axes]
    for ax, (model, cm) in zip(axes, matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Pred Pos', 'Pred Neg'], yticklabels=['Act Pos', 'Act Neg'])
        ax.set_title(model)
    plt.tight_layout()
    plt.savefig("plot_confusion_matrices.png")
    plt.close()

# --- MAIN ---
def main():
    metrics_data = []
    confusion_data = {}
    roc_data = {}
    
    print("Reading CSV files...")
    for model, csv_file in FILES.items():
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                
                # Metrics
                is_neg = df['image'].str.lower().str.contains('negative') | df['image'].str.lower().str.contains('empty')
                pos_df = df[~is_neg]
                recall = 0
                avg_conf = 0
                if len(pos_df) > 0:
                    hits = pos_df['found_people'].notna() & (pos_df['found_people'] != "")
                    recall = (hits.sum() / len(pos_df)) * 100
                    if 'avg_confidence' in pos_df.columns:
                        confs = pos_df['avg_confidence'].apply(clean_confidence)
                        valid = confs[confs > 0]
                        if len(valid) > 0: avg_conf = valid.mean()
                
                metrics_data.append({
                    "Model": model, "Recall": recall, "Avg Conf": avg_conf, "Time": df['time'].mean()
                })
                
                # Confusion Matrix
                confusion_data[model] = calculate_confusion_matrix(df)
                
                # ROC Data (Requires 'raw_matches' column)
                if 'raw_matches' in df.columns:
                    y_true, y_scores = get_roc_data(df)
                    roc_data[model] = (y_true, y_scores)
                
            except Exception as e:
                print(f"Skipping {model}: {e}")

    if not metrics_data:
        print("No data found!")
        return

    print("Generating Plots...")
    plot_bar_comparison(metrics_data)
    plot_efficiency_frontier(metrics_data)
    plot_confusion_matrices(confusion_data)
    
    if roc_data:
        print("Generating ROC Curve...")
        plot_roc_curve(roc_data)
    else:
        print("Skipping ROC (No 'raw_matches' column found. Re-run pipeline with updated utils.py)")
        
    print("All plots saved.")

if __name__ == "__main__":
    main()