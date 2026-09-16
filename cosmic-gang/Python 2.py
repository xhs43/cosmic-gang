import os
import pandas as pd
import numpy as np
from transit_detection_pipeline import TransitDetector

def process_all_sessions(data_dir="processed_lightcurves"):
    detector = TransitDetector()
    results = []

    if not os.path.exists(data_dir):
        print(f"المجلد {data_dir} غير موجود.")
        return

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path)
            
            # استكشاف أعمدة الوقت والشدة الضوئية ديناميكياً
            time_col = [c for c in df.columns if 'time' in c.lower() or 'bjd' in c.lower()][0]
            flux_col = [c for c in df.columns if 'flux' in c.lower() or 'mag' in c.lower()][0]
            
            res = detector.detect_transit(df[time_col].values, df[flux_col].values)
            res['filename'] = file_name
            results.append(res)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("FINAL_DETECTION_SUMMARY.csv", index=False)
    print("تم معالجة جميع ملفات الرصد بنجاح وحفظ النتيجة في FINAL_DETECTION_SUMMARY.csv")

if __name__ == "__main__":
    process_all_sessions()