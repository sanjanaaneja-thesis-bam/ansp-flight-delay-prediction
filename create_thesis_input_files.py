"""Build hub-specific input files for June-July 2023 analysis."""

import pandas as pd
import numpy as np
import os

LARGE_HUBS = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MCO', 'LAS', 'PHX', 'MIA',
              'SEA', 'IAH', 'JFK', 'EWR', 'FLL', 'MSP', 'SFO', 'DTW', 'BOS', 'SLC',
              'PHL', 'BWI', 'TPA', 'SAN', 'LGA', 'MDW', 'BNA', 'IAD', 'DCA', 'AUS']

MEDIUM_HUBS = ['DAL', 'HNL', 'PDX', 'HOU', 'RSW', 'STL', 'SMF', 'SJU', 'RDU', 'MSY', 
               'OAK', 'SNA', 'MCI', 'SAT', 'SJC']

SMALL_HUBS = ['OKC', 'RIC', 'GEG', 'MYR', 'SRQ', 'SDF', 'GRR', 'ELP', 'BUF', 'KOA',
              'SAV', 'TUS', 'SFB', 'PNS', 'PVD']

ALL_HUBS = LARGE_HUBS + MEDIUM_HUBS + SMALL_HUBS

# Airport timezone offsets from UTC (for all 60 airports)
# NOTE: June-July 2023 is DST period, using daylight saving time offsets
AIRPORT_TIMEZONES = {
    # Eastern Daylight Time (UTC-4) - 27 airports
    'ATL': -4, 'CLT': -4, 'MCO': -4, 'MIA': -4, 'FLL': -4, 'DTW': -4, 'BOS': -4,
    'PHL': -4, 'BWI': -4, 'TPA': -4, 'LGA': -4, 'IAD': -4, 'DCA': -4,
    'RDU': -4, 'RIC': -4, 'MYR': -4, 'SRQ': -4, 'SDF': -4, 'BUF': -4,
    'SAV': -4, 'SFB': -4, 'PNS': -4, 'PVD': -4, 'EWR': -4, 'JFK': -4, 'RSW': -4, 'GRR': -4,
    # Central Daylight Time (UTC-5) - 14 airports
    'DFW': -5, 'ORD': -5, 'IAH': -5, 'MSP': -5, 'MDW': -5, 'AUS': -5, 'BNA': -5, 'MSY': -5,
    'DAL': -5, 'HOU': -5, 'STL': -5, 'MCI': -5, 'SAT': -5, 'OKC': -5,
    # Mountain Daylight Time (UTC-6) - 3 airports
    'DEN': -6, 'SLC': -6, 'ELP': -6,
    # Mountain Standard Time (UTC-7, Arizona doesn't observe DST) - 2 airports
    'PHX': -7, 'TUS': -7,
    # Pacific Daylight Time (UTC-7) - 11 airports
    'LAX': -7, 'LAS': -7, 'SEA': -7, 'SFO': -7, 'SAN': -7,
    'PDX': -7, 'SMF': -7, 'OAK': -7, 'SNA': -7, 'SJC': -7, 'GEG': -7,
    # Hawaii Standard Time (UTC-10, no DST) - 2 airports
    'HNL': -10, 'KOA': -10,
    # Atlantic Standard Time (UTC-4, Puerto Rico doesn't observe DST) - 1 airport
    'SJU': -4,
}

# Expected 57 columns in final output (input_data.csv has 56, we drop MKT_CARRIER and add 4 raw time fields)
FINAL_COLUMNS = [
    # BTS raw data (25 columns - includes raw time fields)
    'MONTH', 'DAY_OF_MONTH', 'FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST',
    'CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME',  # Raw time fields in HHMM format
    'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15',
    'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY',
    
    # Traffic metrics (6 columns) - placeholders for medium/small
    'CRS_DEP_1hrpre', 'CRS_DEP_1hrpost', 'DEP_1hrpre_num', 'DEP_1hrpost_num',
    'Arr_1hrpre_num', 'Arr_1hrpost_num',
    
    # Weather data (8 columns)
    'max_temp_f', 'min_temp_f', 'max_dewpoint_f', 'min_dewpoint_f',
    'precip_in', 'avg_wind_speed_kts', 'snow_in', 'avg_feel',
    
    # FAA classification (1 column)
    'FAA_class',
    
    # Turnaround metrics - FIRST 4 (placeholders for medium/small)
    'scheduled_Turnarnd', 'Actual_Turnarnd', 'Diff_in_turnarnd', 'longTurnaround',
    
    # Time conversions (7 columns) - calculated from raw BTS time fields
    'Scheduled_DEP', 'Scheduled_DEP_EST', 'Actual_DEP_dt_EST',
    'Scheduled_ARR_Local', 'Actual_ARR_dt_Local', 'Scheduled_ARR_EST', 'Actual_ARR_dt_EST',
    
    # Turnaround metrics - LAST 5 (placeholders for medium/small)
    'late_airjet_when_turnaround_within_180', 'affected_turnaround_lessthan120',
    'affected_turnaround_lessthan90', 'affected_turnaround_lessthan60',
    'affected_turnaround_lessthan45',
    
    # Day of week (1 column)
    'day_of_week',
]

def parse_hhmm_to_time(hhmm_value):
    """Convert HHMM integer (e.g. 1430) to a time object, or None if invalid."""
    if pd.isna(hhmm_value):
        return None
    
    try:
        hhmm_int = int(float(hhmm_value))
        
        hours = hhmm_int // 100
        minutes = hhmm_int % 100
        
        if hours >= 24 or minutes >= 60:
            return None
        
        return pd.Timestamp(f"1900-01-01 {hours:02d}:{minutes:02d}:00").time()
    except (ValueError, TypeError):
        return None

def build_datetime(fl_date, time_obj):
    """Combine flight date and time object into a datetime."""

    if pd.isna(fl_date) or time_obj is None:
        return pd.NaT
    
    try:
        if not isinstance(fl_date, pd.Timestamp):
            fl_date = pd.to_datetime(fl_date)
        
        return pd.Timestamp.combine(fl_date.date(), time_obj)
    except Exception:
        return pd.NaT

def convert_to_est(local_dt, airport_tz_offset):
    if pd.isna(local_dt):
        return pd.NaT
    
    try:
        # Convert local to UTC by subtracting the airport offset
        # Example: Pacific Daylight (UTC-7) 10:00 → UTC 17:00 (add 7 hours)
        utc_dt = local_dt - pd.Timedelta(hours=airport_tz_offset)
        
        # Convert UTC to EDT (UTC-4) during DST period
        # Example: UTC 17:00 → EDT 13:00 (subtract 4 hours)
        est_dt = utc_dt + pd.Timedelta(hours=-4)
        
        return est_dt
    except Exception:
        return pd.NaT

def calculate_time_columns(df):
    """Derive 7 datetime columns (local and EST) from raw BTS HHMM time fields."""
    required_cols = ['CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME',
                     'FL_DATE', 'ORIGIN', 'DEST']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"WARNING: Missing required columns: {missing}")
        return df

    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')

    df['_crs_dep_time_obj'] = df['CRS_DEP_TIME'].apply(parse_hhmm_to_time)
    df['_dep_time_obj'] = df['DEP_TIME'].apply(parse_hhmm_to_time)

    # Step 1: build departure datetimes
    df['Scheduled_DEP'] = df.apply(
        lambda row: build_datetime(row['FL_DATE'], row['_crs_dep_time_obj']), axis=1
    )

    # Actual departure: build from FL_DATE + DEP_TIME, then handle midnight crossing
    # If flight was scheduled late evening (CRS >= 1800) but departed early morning
    # (DEP < 0600), the actual departure crossed midnight → add 1 day.
    df['_actual_dep_local'] = df.apply(
        lambda row: build_datetime(row['FL_DATE'], row['_dep_time_obj']), axis=1
    )
    _crs_dep_int = pd.to_numeric(df['CRS_DEP_TIME'], errors='coerce')
    _dep_int = pd.to_numeric(df['DEP_TIME'], errors='coerce')
    _dep_cross_midnight = (
        _crs_dep_int.notna() & _dep_int.notna() &
        (_crs_dep_int >= 1800) & (_dep_int < 600)
    )
    df.loc[_dep_cross_midnight & df['_actual_dep_local'].notna(),
           '_actual_dep_local'] += pd.Timedelta(days=1)
    print(f"      Actual departures +1 day (midnight crossing): {_dep_cross_midnight.sum():,}")

    df['Scheduled_DEP_EST'] = df.apply(
        lambda row: convert_to_est(row['Scheduled_DEP'], AIRPORT_TIMEZONES.get(row['ORIGIN'], -4)),
        axis=1
    )
    df['Actual_DEP_dt_EST'] = df.apply(
        lambda row: convert_to_est(row['_actual_dep_local'], AIRPORT_TIMEZONES.get(row['ORIGIN'], -4)),
        axis=1
    )

    # Step 2: derive arrival datetimes (DEP_EST + elapsed, same as authors)
    _crs_elapsed = pd.to_numeric(df['CRS_ELAPSED_TIME'], errors='coerce')
    _act_elapsed = pd.to_numeric(df['ACTUAL_ELAPSED_TIME'], errors='coerce')

    df['Scheduled_ARR_EST'] = df['Scheduled_DEP_EST'] + pd.to_timedelta(_crs_elapsed, unit='m')
    df['Actual_ARR_dt_EST'] = df['Actual_DEP_dt_EST'] + pd.to_timedelta(_act_elapsed, unit='m')

    # Step 3: convert arrivals to destination local time
    # ARR_Local = ARR_EST + (4 + dest_offset) hours
    df['Scheduled_ARR_Local'] = df.apply(
        lambda row: (row['Scheduled_ARR_EST'] + pd.Timedelta(hours=4 + AIRPORT_TIMEZONES.get(row['DEST'], -4)))
                     if pd.notna(row['Scheduled_ARR_EST']) else pd.NaT,
        axis=1
    )
    df['Actual_ARR_dt_Local'] = df.apply(
        lambda row: (row['Actual_ARR_dt_EST'] + pd.Timedelta(hours=4 + AIRPORT_TIMEZONES.get(row['DEST'], -4)))
                     if pd.notna(row['Actual_ARR_dt_EST']) else pd.NaT,
        axis=1
    )

    temp_cols = ['_crs_dep_time_obj', '_dep_time_obj', '_actual_dep_local']
    df = df.drop(columns=temp_cols)

    return df

def create_large_hub_file():
    df = pd.read_csv('input_data.csv')
    print(f"input_data.csv: {len(df):,} rows x {df.shape[1]} cols")

    if 'MKT_CARRIER' in df.columns:
        df = df.drop(columns=['MKT_CARRIER'])

    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df = df[(df['FL_DATE'] >= '2023-06-01') & (df['FL_DATE'] <= '2023-07-31')].copy()
    print(f"  After date filter: {len(df):,} rows")
    df = df[df['ORIGIN'].isin(LARGE_HUBS)].copy()
    print(f"  After ORIGIN filter: {len(df):,} rows")
    df = df[df['DEST'].isin(ALL_HUBS)].copy()
    print(f"  After DEST filter: {len(df):,} rows")
    
    raw_time_cols = ['CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME']
    has_raw_times = all(col in df.columns for col in raw_time_cols)

    if has_raw_times:
        datetime_cols_to_drop = ['Scheduled_DEP', 'Scheduled_DEP_EST', 'Actual_DEP_dt_EST',
                                  'Scheduled_ARR_Local', 'Actual_ARR_dt_Local',
                                  'Scheduled_ARR_EST', 'Actual_ARR_dt_EST']
        existing_datetime_cols = [col for col in datetime_cols_to_drop if col in df.columns]
        if existing_datetime_cols:
            df = df.drop(columns=existing_datetime_cols)
        df = calculate_time_columns(df)
    else:
        print("NOTE: No raw time fields in input_data.csv. keeping authors' datetime columns")

    # Fix flights where ARR_EST < DEP_EST (PHX->TUS timezone quirk in authors' data)
    for dt_col in ['Scheduled_DEP_EST', 'Scheduled_ARR_EST',
                    'Actual_DEP_dt_EST', 'Actual_ARR_dt_EST']:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')

    n_sched = 0
    n_act = 0
    if {'Scheduled_DEP_EST', 'Scheduled_ARR_EST', 'CRS_ELAPSED_TIME'}.issubset(df.columns):
        sched_bad = (df['Scheduled_ARR_EST'].notna() &
                     df['Scheduled_DEP_EST'].notna() &
                     (df['Scheduled_ARR_EST'] < df['Scheduled_DEP_EST']))
        n_sched = int(sched_bad.sum())
        if n_sched > 0:
            _elapsed = pd.to_numeric(df.loc[sched_bad, 'CRS_ELAPSED_TIME'], errors='coerce')
            df.loc[sched_bad, 'Scheduled_ARR_EST'] = (
                df.loc[sched_bad, 'Scheduled_DEP_EST']
                + pd.to_timedelta(_elapsed, unit='m')
            )
            print(f"  Fixed {n_sched} scheduled arrivals (ARR_EST < DEP_EST)")

    if {'Actual_DEP_dt_EST', 'Actual_ARR_dt_EST', 'ACTUAL_ELAPSED_TIME'}.issubset(df.columns):
        act_bad = (df['Actual_ARR_dt_EST'].notna() &
                   df['Actual_DEP_dt_EST'].notna() &
                   (df['Actual_ARR_dt_EST'] < df['Actual_DEP_dt_EST']))
        n_act = int(act_bad.sum())
        if n_act > 0:
            _elapsed = pd.to_numeric(df.loc[act_bad, 'ACTUAL_ELAPSED_TIME'], errors='coerce')
            df.loc[act_bad, 'Actual_ARR_dt_EST'] = (
                df.loc[act_bad, 'Actual_DEP_dt_EST']
                + pd.to_timedelta(_elapsed, unit='m')
            )
            print(f"  Fixed {n_act} actual arrivals (ARR < DEP)")

    # Align column names with FINAL_COLUMNS
    df = df.rename(columns={
        'Scheduled_ARR_Ori': 'Scheduled_ARR_Local',
        'Actual_ARR_dt_Ori': 'Actual_ARR_dt_Local',
    })
    missing_cols = [col for col in FINAL_COLUMNS if col not in df.columns]
    for col in missing_cols:
        df[col] = np.nan
    df = df[FINAL_COLUMNS].copy()

    output_file = 'input_data_large_filtered.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}: {os.path.getsize(output_file)/1024**2:.1f} MB, {len(df):,} rows x {df.shape[1]} cols")

    return df

def create_medium_small_hub_file(hub_category, airport_list):
    df_june = pd.read_csv('BTS_ONTIME_REPORTING_June_2023.csv')
    df_july = pd.read_csv('BTS_ONTIME_REPORTING_July_2023.csv')
    print(f"BTS: June {len(df_june):,} rows, July {len(df_july):,} rows")

    df = pd.concat([df_june, df_july], ignore_index=True)
    df = df[df['ORIGIN'].isin(airport_list)].copy()
    print(f"  After ORIGIN filter: {len(df):,} rows")
    df = df[df['DEST'].isin(ALL_HUBS)].copy()
    print(f"  After DEST filter: {len(df):,} rows")
    bts_column_map = {
        'OP_UNIQUE_CARRIER': 'OP_CARRIER',
        'DAY_OF_WEEK': 'day_of_week',
    }
    df = df.rename(columns=bts_column_map)
    
    df_weather = pd.read_csv('weather_data/all_airports_weather_combined.csv')
    df_weather = df_weather.rename(columns={'IATA': 'ORIGIN', 'day': 'FL_DATE'})
    df_weather['FL_DATE'] = pd.to_datetime(df_weather['FL_DATE'], errors='coerce')
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    if 'station' in df_weather.columns:
        df_weather = df_weather.drop(columns=['station'])
    before_merge = len(df)
    df = df.merge(df_weather, on=['ORIGIN', 'FL_DATE'], how='left')
    print(f"  Weather join: {len(df):,} rows (was {before_merge:,}), missing rate: {df['max_temp_f'].isna().mean():.2%}")

    df_faa = pd.read_csv('FAA_selected_airports_sample.csv')
    df_faa = df_faa[['Airport_Code', 'Hub_Category']].rename(columns={
        'Airport_Code': 'ORIGIN',
        'Hub_Category': 'FAA_class'
    })
    
    df = df.merge(df_faa, on='ORIGIN', how='left')
    print(f"  FAA_class missing rate: {df['FAA_class'].isna().mean():.2%}")

    # Add placeholder columns for traffic metrics (6 columns)
    for col in ['CRS_DEP_1hrpre', 'CRS_DEP_1hrpost', 'DEP_1hrpre_num',
                'DEP_1hrpost_num', 'Arr_1hrpre_num', 'Arr_1hrpost_num']:
        df[col] = np.nan

    # Add placeholder columns for turnaround metrics (9 columns)
    for col in ['scheduled_Turnarnd', 'Actual_Turnarnd', 'Diff_in_turnarnd',
                'longTurnaround', 'late_airjet_when_turnaround_within_180',
                'affected_turnaround_lessthan120', 'affected_turnaround_lessthan90',
                'affected_turnaround_lessthan60', 'affected_turnaround_lessthan45']:
        df[col] = np.nan

    df = calculate_time_columns(df)

    missing_cols = [col for col in FINAL_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")
        for col in missing_cols:
            df[col] = np.nan
    df = df[FINAL_COLUMNS].copy()
    if df.shape[1] != 57:
        print(f"  WARNING: Expected 57 columns, got {df.shape[1]}")

    output_file = f'input_data_{hub_category.lower()}_filtered.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}: {os.path.getsize(output_file)/1024**2:.1f} MB, {len(df):,} rows x {df.shape[1]} cols")
    
    return df

def main():
    try:
        create_large_hub_file()
    except Exception as e:
        print(f"Error creating large hub file: {e}")

    try:
        create_medium_small_hub_file('Medium', MEDIUM_HUBS)
    except Exception as e:
        print(f"Error creating medium hub file: {e}")

    try:
        create_medium_small_hub_file('Small', SMALL_HUBS)
    except Exception as e:
        print(f"Error creating small hub file: {e}")

if __name__ == '__main__':
    main()
