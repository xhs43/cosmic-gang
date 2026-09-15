import os
import pandas as pd
import numpy as np
from astropy.time import Time
import warnings
import sys

# Fix for windows console printing Arabic
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

def process_exoplanet_data():
    print("بدء معالجة وفهرسة البيانات...\n" + "-"*40)
    
    # Files to check
    files_to_check = ['dataset_index.csv', 'observations.csv', 'observations.json']
    df = None
    file_used = ""
    
    for f in files_to_check:
        if os.path.exists(f):
            print(f"[+] تم العثور على ملف البيانات: {f}")
            if f.endswith('.csv'):
                df = pd.read_csv(f)
            else:
                df = pd.read_json(f)
            file_used = f
            # If dataset_index is found, prefer it as it contains the 1741 items mentioned
            if f == 'dataset_index.csv':
                break
                
    if df is None:
        raise FileNotFoundError("[-] خطأ: لم يتم العثور على أي من ملفات البيانات.")

    # Standardize column names
    df.columns = [str(c).upper().strip() for c in df.columns]
    
    # Identify key columns
    target_col = 'TARGET_NAME' if 'TARGET_NAME' in df.columns else 'OBJECT' if 'OBJECT' in df.columns else 'TARGET' if 'TARGET' in df.columns else None
    filter_col = 'FILTER' if 'FILTER' in df.columns else None
    date_col = 'DATE_OBS' if 'DATE_OBS' in df.columns else 'DATE-OBS' if 'DATE-OBS' in df.columns else None
    
    if target_col is None:
        raise ValueError("[-] خطأ: لا يوجد عمود باسم الهدف في البيانات.")
        
    df[target_col] = df[target_col].fillna('Unknown').astype(str).str.strip()

    # Isolate calibration frames
    is_calib_target = df[target_col].str.lower().isin(['dark', 'calibration', 'bias', 'flat', 'opaque', 'dark-c-'])
    is_calib_target = is_calib_target | df[target_col].str.lower().str.contains('dark')
    
    is_calib_filter = pd.Series(False, index=df.index)
    if filter_col:
        df[filter_col] = df[filter_col].fillna('Unknown').astype(str).str.strip()
        is_calib_filter = df[filter_col].str.lower().isin(['opaque', 'dark'])
        
    calib_mask = is_calib_target | is_calib_filter
    
    df['FRAME_TYPE'] = np.where(calib_mask, 'Calibration', 'Science')
    
    # Validate the 8 targets
    valid_targets = ['CoRoT-2', 'Qatar-1', 'TrES-3', 'TrES-5', 'HAT-P-10', 'WASP-2', 'WASP-10', 'TRES-1']
    target_mapping = {t.lower(): t for t in valid_targets}
    
    df['normalized_target'] = df[target_col].str.lower()
    
    science_mask = df['FRAME_TYPE'] == 'Science'
    valid_science_mask = science_mask & df['normalized_target'].isin(target_mapping.keys())
    
    df.loc[valid_science_mask, target_col] = df.loc[valid_science_mask, 'normalized_target'].map(target_mapping)
    
    df_clean = df[calib_mask | valid_science_mask].copy()
    df_clean = df_clean.drop(columns=['normalized_target'])
    
    # Standardize time (DATE-OBS and BJD)
    if date_col:
        # Convert date to standard pandas datetime, UTC
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce', utc=True)
        df_clean = df_clean.dropna(subset=[date_col])
        
        # Compute BJD if not present (we'll compute JD as placeholder)
        if 'BJD' not in df_clean.columns and 'TIME_JD' not in df_clean.columns:
            valid_dates = df_clean[date_col].dropna()
            if not valid_dates.empty:
                t = Time(valid_dates.tolist(), scale='utc')
                df_clean.loc[valid_dates.index, 'BJD'] = t.jd
        elif 'TIME_JD' in df_clean.columns:
            df_clean['BJD'] = pd.to_numeric(df_clean['TIME_JD'], errors='coerce')
        else:
            df_clean['BJD'] = pd.to_numeric(df_clean['BJD'], errors='coerce')
            
        # Sort by target and time
        df_clean = df_clean.sort_values(by=[target_col, date_col])
        
        # Calculate time difference
        df_clean['TIME_DIFF_HOURS'] = df_clean.groupby(target_col)[date_col].diff().dt.total_seconds() / 3600.0
        
        # Group into sessions (difference > 8 hours)
        is_new_session = df_clean['TIME_DIFF_HOURS'] > 8.0
        df_clean['SESSION_ID'] = is_new_session.groupby(df_clean[target_col]).cumsum() + 1
        
        df_clean['SESSION_ID'] = df_clean[target_col] + "_Session_" + df_clean['SESSION_ID'].astype(str)
        df_clean.loc[df_clean['FRAME_TYPE'] == 'Calibration', 'SESSION_ID'] = 'Calibration_Session'

    # Save cleaned output
    output_filename = "CLEANED_DATA.csv"
    front_cols = [target_col, 'FRAME_TYPE', date_col, 'BJD', 'SESSION_ID']
    front_cols = [c for c in front_cols if c in df_clean.columns]
    other_cols = [c for c in df_clean.columns if c not in front_cols]
    df_clean = df_clean[front_cols + other_cols]
    
    df_clean.to_csv(output_filename, index=False)
    print(f"[+] تم الانتهاء من التنظيف! حُفظ الناتج في: {output_filename}")
    
    # Statistical report
    print("\n" + "="*45)
    print(" 📊 التقرير الإحصائي المفصل لبيانات التلسكوب")
    print("="*45)
    
    total_original = len(df)
    total_clean = len(df_clean)
    calib_count = len(df_clean[df_clean['FRAME_TYPE'] == 'Calibration'])
    science_count = len(df_clean[df_clean['FRAME_TYPE'] == 'Science'])
    
    print(f"إجمالي الأطر (قبل التنظيف):   {total_original:,} إطار")
    print(f"إجمالي الأطر (بعد التنظيف):    {total_clean:,} إطار")
    print(f"الأطر المستبعدة (غير صالحة):   {total_original - total_clean:,} إطار\n")
    
    print(f"أطر المعايرة (Dark/Opaque):   {calib_count:,} إطار")
    print(f"الأطر العلمية (Science):       {science_count:,} إطار\n")
    
    print("تفصيل الأطر العلمية لكل كوكب:")
    print("-" * 30)
    planet_counts = df_clean[df_clean['FRAME_TYPE'] == 'Science'][target_col].value_counts()
    
    if planet_counts.empty:
        print("لا يوجد إطارات علمية صالحة للأهداف المحددة.")
    else:
        if 'SESSION_ID' in df_clean.columns:
            sessions_counts = df_clean[df_clean['FRAME_TYPE'] == 'Science'].groupby(target_col)['SESSION_ID'].nunique()
            for planet in valid_targets:
                if planet in planet_counts:
                    frames = planet_counts[planet]
                    sessions = sessions_counts[planet]
                    print(f" 🔹 {planet:<10}: {frames:>4} إطار (موزعة على {sessions} جلسات رصد)")
        else:
            for planet in valid_targets:
                if planet in planet_counts:
                    frames = planet_counts[planet]
                    print(f" 🔹 {planet:<10}: {frames:>4} إطار")
            
    print("="*45)

if __name__ == "__main__":
    process_exoplanet_data()
