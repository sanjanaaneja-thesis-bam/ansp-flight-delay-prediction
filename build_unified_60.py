#Merge Large/Medium/Small filtered files into input_data_unified_60.csv.

import pandas as pd
import numpy as np
import os
import sys

if sys.platform == 'win32':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

UNIFIED_COLUMNS = [
    # identifiers
    'MONTH', 'DAY_OF_MONTH', 'FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST',
    # delay
    'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15',
    # flight
    'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
    # delay causes
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY',
    # traffic
    'DEP_1hrpre_num', 'DEP_1hrpost_num', 'Arr_1hrpre_num', 'Arr_1hrpost_num',
    # weather
    'max_temp_f', 'min_temp_f', 'max_dewpoint_f', 'min_dewpoint_f',
    'precip_in', 'avg_wind_speed_kts', 'snow_in', 'avg_feel',
    # FAA class
    'FAA_class',
    # turnaround
    'scheduled_Turnarnd', 'Actual_Turnarnd', 'Diff_in_turnarnd', 'longTurnaround',
    'late_airjet_when_turnaround_within_180',
    'affected_turnaround_lessthan120', 'affected_turnaround_lessthan90',
    'affected_turnaround_lessthan60', 'affected_turnaround_lessthan45',
    # datetimes
    'Scheduled_DEP', 'Scheduled_DEP_EST', 'Actual_DEP_dt_EST',
    'Scheduled_ARR_Local', 'Actual_ARR_dt_Local', 'Scheduled_ARR_EST', 'Actual_ARR_dt_EST',
    # day of week
    'day_of_week',
]

DOW_MAP = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday',
           5: 'Friday', 6: 'Saturday', 7: 'Sunday'}

def main():
    dfs = []
    for name, filepath in [('Large', 'input_data_large_filtered.csv'),
                            ('Medium', 'input_data_medium_filtered.csv'),
                            ('Small', 'input_data_small_filtered.csv')]:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  Raw: {len(df):,} rows x {df.shape[1]} cols")

        if 'FAA_class' in df.columns:
            df['FAA_class'] = df['FAA_class'].replace({'Lrg': 'Large'})

        if df['day_of_week'].dtype in ['int64', 'float64']:
            df['day_of_week'] = df['day_of_week'].map(DOW_MAP)

        missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
        if missing:
            print(f"  WARNING: Missing columns (NaN): {missing}")
            for c in missing:
                df[c] = np.nan

        df = df[UNIFIED_COLUMNS].copy()
        dfs.append(df)

    unified = pd.concat(dfs, ignore_index=True)
    print(f"\nUnified: {len(unified):,} rows x {unified.shape[1]} cols")

    # NaN turnaround flags mean no matched previous leg in BTS; treat as 0
    turnaround_flags = ['affected_turnaround_lessthan60', 'affected_turnaround_lessthan45',
                        'affected_turnaround_lessthan90', 'affected_turnaround_lessthan120',
                        'longTurnaround', 'late_airjet_when_turnaround_within_180']
    for col in turnaround_flags:
        if col in unified.columns:
            before = unified[col].isna().sum()
            unified[col] = unified[col].fillna(0).astype(int)
            print(f"  {col}: {before:,} NaN -> 0")

    ALL_60 = set(pd.read_csv('FAA_selected_airports_sample.csv')['Airport_Code'])
    missing_origins = ALL_60 - set(unified['ORIGIN'].unique())
    if missing_origins:
        print(f"  WARNING: Missing ORIGIN airports: {missing_origins}")

    for cat in ['Large', 'Medium', 'Small']:
        n = (unified['FAA_class'] == cat).sum()
        n_apt = unified[unified['FAA_class'] == cat]['ORIGIN'].nunique()
        print(f"  {cat}: {n:,} flights, {n_apt} airports")

    unified['FL_DATE'] = pd.to_datetime(unified['FL_DATE'], errors='coerce')
    print(f"Date range: {unified['FL_DATE'].min().date()} to {unified['FL_DATE'].max().date()}")

    output = 'input_data_unified_60.csv'
    unified.to_csv(output, index=False)
    size_mb = os.path.getsize(output) / 1024**2
    print(f"\nSaved: {output} ({size_mb:.1f} MB, {len(unified):,} rows)")

if __name__ == '__main__':
    main()
