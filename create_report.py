import pandas as pd
import os
from docx import Document

# REMOVED "DeepFace" from this list
FILES = {
    "FaceNet": "results_facenet.csv",
    "ArcFace": "results_arcface.csv",
    "CosFace": "results_cosface.csv",
    "SphereFace": "results_sphereface.csv"
}

def generate_report():
    print("Generating Final Report...")
    doc = Document()
    doc.add_heading('Face Recognition Benchmark', 0)
    
    # Summary Table
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = 'Model'
    hdr[1].text = 'Recall (Accuracy)'
    hdr[2].text = 'False Alarm Rate'
    hdr[3].text = 'Avg Time (s)'
    
    for model, csv_file in FILES.items():
        row = table.add_row().cells
        row[0].text = model
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                
                # 1. Positive Images (Recall)
                pos_df = df[df['type'] == 'Positive']
                pos_acc = 0.0
                if len(pos_df) > 0:
                    pos_hits = len(pos_df[pos_df['success'] == True])
                    pos_acc = round((pos_hits / len(pos_df) * 100), 1)
                
                # 2. Negative Images (False Alarms)
                neg_df = df[df['type'] == 'Negative']
                false_alarm = 0.0
                if len(neg_df) > 0:
                    ghosts = len(neg_df[neg_df['success'] == False])
                    false_alarm = round((ghosts / len(neg_df) * 100), 1)
                
                row[1].text = f"{pos_acc}%"
                row[2].text = f"{false_alarm}%"
                row[3].text = str(round(df['time'].mean(), 4))
            except Exception as e:
                row[1].text = "Error"
                print(f"Skipping {model} due to error: {e}")
        else:
            row[1].text = "-"
            row[2].text = "-"
            row[3].text = "Not Run"

    doc.save('Final_Benchmark_Report.docx')
    print("Done! Saved as 'Final_Benchmark_Report.docx'")

if __name__ == "__main__":
    generate_report()