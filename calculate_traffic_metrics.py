"""Compute 6 traffic metrics from full BTS.
  For each flight in the Medium/Small hub filtered files, compute:
    - CRS_DEP_1hrpre, CRS_DEP_1hrpost: count of scheduled departures at ORIGIN
      in a 1-hour window before/after this flight's scheduled departure
    - DEP_1hrpre_num, DEP_1hrpost_num: actual departures at ORIGIN
    - Arr_1hrpre_num, Arr_1hrpost_num: actual arrivals at DEST
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

if sys.platform == 'win32':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

MEDIUM_HUBS = ['DAL','HNL','PDX','HOU','RSW','STL','SMF','SJU','RDU','MSY',
               'OAK','SNA','MCI','SAT','SJC']
SMALL_HUBS = ['OKC','RIC','GEG','MYR','SRQ','SDF','GRR','ELP','BUF','KOA',
              'SAV','TUS','SFB','PNS','PVD']
LARGE_HUBS = ['ATL','DFW','DEN','ORD','LAX','CLT','MCO','LAS','PHX','MIA',
              'SEA','IAH','JFK','EWR','FLL','MSP','SFO','DTW','BOS','SLC',
              'PHL','BWI','TPA','SAN','LGA','MDW','BNA','IAD','DCA','AUS']
ALL_60 = set(LARGE_HUBS + MEDIUM_HUBS + SMALL_HUBS)
MED_SMALL = set(MEDIUM_HUBS + SMALL_HUBS)

# Local datetime construction

def hhmm_to_minutes(hhmm_series):
    """Convert HHMM integer series to minutes-since-midnight. NaN for invalid."""
    h = pd.to_numeric(hhmm_series, errors='coerce')
    hours = h // 100
    minutes = h % 100
    # Validate: hours in [0,24), minutes in [0,60)
    mask = (h.notna()) & (hours < 24) & (minutes < 60)
    result = pd.Series(np.nan, index=h.index)
    result[mask] = (hours[mask] * 60 + minutes[mask]).astype(float)
    return result

def build_local_datetimes(df):
    df = df.copy()
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')

    # Convert HHMM to minutes-since-midnight
    crs_dep_min = hhmm_to_minutes(df['CRS_DEP_TIME'])
    dep_min = hhmm_to_minutes(df['DEP_TIME'])
    crs_arr_min = hhmm_to_minutes(df['CRS_ARR_TIME'])
    arr_min = hhmm_to_minutes(df['ARR_TIME'])

    base_date = df['FL_DATE']

    # Scheduled departure: always same day as FL_DATE 
    df['_dep_local_sched'] = base_date + pd.to_timedelta(crs_dep_min, unit='m')

    # Scheduled arrival: +1 day if CRS_ARR < CRS_DEP (crosses midnight) 
    sched_arr_offset = (crs_arr_min < crs_dep_min).fillna(False).astype(int)
    df['_arr_local_sched'] = (base_date
                              + pd.to_timedelta(sched_arr_offset, unit='D')
                              + pd.to_timedelta(crs_arr_min, unit='m'))

    # Actual departure: +1 day if scheduled late evening and actual is early morning 
    dep_day_offset = ((crs_dep_min >= 1080) & (dep_min < 360)).fillna(False).astype(int)
    df['_dep_local_actual'] = (base_date
                               + pd.to_timedelta(dep_day_offset, unit='D')
                               + pd.to_timedelta(dep_min, unit='m'))

    #  Actual arrival: gets dep_day_offset, plus 1 more if ARR < DEP 
    arr_extra_offset = (arr_min < dep_min).fillna(False).astype(int)
    arr_total_offset = dep_day_offset + arr_extra_offset
    df['_arr_local_actual'] = (base_date
                               + pd.to_timedelta(arr_total_offset, unit='D')
                               + pd.to_timedelta(arr_min, unit='m'))

    return df

# Traffic metric counting

def count_window_per_airport(df, airport_col, time_col, window_start_h, window_end_h):
    """
    For each row in df, count how many OTHER rows at the same airport_col value
    have time_col within [current_time + window_start_h, current_time + window_end_h).
    The window is half-open: [start, end).
    The current row is excluded from its own count.
    Rows whose own ref time is NaN (cancelled/diverted flights when counting
    against actual-time columns) receive NaN, not 0 as a zero here would be a
    false absence of traffic rather than a missing observation.
    """
    counts = np.full(len(df), np.nan, dtype=float)
    df = df.reset_index(drop=True)
    ref_times = df[time_col].values  # datetime64[ns]

    grouped = df.groupby(airport_col, sort=False)

    for airport, group in tqdm(grouped, desc=f"  {time_col[-15:]}@{airport_col}", total=len(grouped)):
        idxs = group.index.values
        times = group[time_col].values
        valid = ~pd.isna(times)

        for i, idx in enumerate(idxs):
            ref = ref_times[idx]
            if pd.isna(ref):
                continue  # leave counts[idx] as NaN
            ws = ref + np.timedelta64(int(window_start_h * 60), 'm')
            we = ref + np.timedelta64(int(window_end_h * 60), 'm')
            in_win = (times >= ws) & (times < we) & valid
            in_win[i] = False  # exclude self
            counts[idx] = float(np.sum(in_win))

    return counts

def compute_all_traffic_metrics(bts_full):
    bts_full = build_local_datetimes(bts_full)

    bts_dep_pool = bts_full[bts_full['ORIGIN'].isin(MED_SMALL)].copy().reset_index(drop=True)
    print(f"DEP pool: {len(bts_dep_pool):,} rows, {bts_dep_pool['ORIGIN'].nunique()} airports")

    bts_dep_pool['CRS_DEP_1hrpre_new'] = count_window_per_airport(
        bts_dep_pool, 'ORIGIN', '_dep_local_sched', -1, 0)
    bts_dep_pool['CRS_DEP_1hrpost_new'] = count_window_per_airport(
        bts_dep_pool, 'ORIGIN', '_dep_local_sched', 0, 1)
    bts_dep_pool['DEP_1hrpre_num_new'] = count_window_per_airport(
        bts_dep_pool, 'ORIGIN', '_dep_local_actual', -1, 0)
    bts_dep_pool['DEP_1hrpost_num_new'] = count_window_per_airport(
        bts_dep_pool, 'ORIGIN', '_dep_local_actual', 0, 1)

    bts_arr_pool = bts_full[bts_full['DEST'].isin(ALL_60)].copy().reset_index(drop=True)
    print(f"ARR pool: {len(bts_arr_pool):,} rows, {bts_arr_pool['DEST'].nunique()} airports")

    bts_arr_pool['Arr_1hrpre_num_new'] = count_window_per_airport(
        bts_arr_pool, 'DEST', '_arr_local_actual', -1, 0)
    bts_arr_pool['Arr_1hrpost_num_new'] = count_window_per_airport(
        bts_arr_pool, 'DEST', '_arr_local_actual', 0, 1)

    return bts_dep_pool, bts_arr_pool

# MERGE BACK TO FILTERED FILES

def make_key(df, fl_date_col='FL_DATE'):
    """Composite merge key: FL_DATE|OP_CARRIER|ORIGIN|DEST|CRS_DEP_TIME."""
    fd = pd.to_datetime(df[fl_date_col], errors='coerce').dt.strftime('%Y-%m-%d')
    carrier = df['OP_CARRIER'].astype(str)
    origin = df['ORIGIN'].astype(str)
    dest = df['DEST'].astype(str)
    cdep = pd.to_numeric(df['CRS_DEP_TIME'], errors='coerce').fillna(-1).astype(int).astype(str)
    return fd + '|' + carrier + '|' + origin + '|' + dest + '|' + cdep

def update_filtered_file(file_path, bts_dep_pool, bts_arr_pool):
    """
    Load a Medium/Small filtered file, overwrite its 6 traffic columns with
    values from the full-BTS pools, and save.
    """
    df = pd.read_csv(file_path, low_memory=False)
    print(f"\nUpdating {file_path}: {len(df):,} rows")

    # Build merge keys
    df['_mkey'] = make_key(df)
    bts_dep_pool = bts_dep_pool.copy()
    bts_dep_pool['_mkey'] = make_key(bts_dep_pool)
    bts_arr_pool = bts_arr_pool.copy()
    bts_arr_pool['_mkey'] = make_key(bts_arr_pool)

    # Drop duplicate keys from pools (keep first occurrence)
    dep_lookup = bts_dep_pool[['_mkey', 'CRS_DEP_1hrpre_new', 'CRS_DEP_1hrpost_new',
                                'DEP_1hrpre_num_new', 'DEP_1hrpost_num_new']].drop_duplicates('_mkey')
    arr_lookup = bts_arr_pool[['_mkey', 'Arr_1hrpre_num_new', 'Arr_1hrpost_num_new']].drop_duplicates('_mkey')

    df = df.merge(dep_lookup, on='_mkey', how='left')
    df = df.merge(arr_lookup, on='_mkey', how='left')

    # Report match rates
    for col_new in ['CRS_DEP_1hrpre_new', 'CRS_DEP_1hrpost_new', 'DEP_1hrpre_num_new',
                     'DEP_1hrpost_num_new', 'Arr_1hrpre_num_new', 'Arr_1hrpost_num_new']:
        matched = df[col_new].notna().sum()
        print(f"    {col_new}: {matched:,}/{len(df):,} matched ({matched/len(df)*100:.1f}%)")

    # Overwrite the 6 traffic columns
    df['CRS_DEP_1hrpre'] = df['CRS_DEP_1hrpre_new']
    df['CRS_DEP_1hrpost'] = df['CRS_DEP_1hrpost_new']
    df['DEP_1hrpre_num'] = df['DEP_1hrpre_num_new']
    df['DEP_1hrpost_num'] = df['DEP_1hrpost_num_new']
    df['Arr_1hrpre_num'] = df['Arr_1hrpre_num_new']
    df['Arr_1hrpost_num'] = df['Arr_1hrpost_num_new']

    # Drop temp columns
    df = df.drop(columns=['_mkey', 'CRS_DEP_1hrpre_new', 'CRS_DEP_1hrpost_new',
                           'DEP_1hrpre_num_new', 'DEP_1hrpost_num_new',
                           'Arr_1hrpre_num_new', 'Arr_1hrpost_num_new'])

    df.to_csv(file_path, index=False)
    print(f"  Saved {file_path}: {len(df):,} rows x {df.shape[1]} cols")

    return df

# MAIN

def main():
    cols = ['MONTH', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST',
            'CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME']
    bts_j = pd.read_csv('BTS_ONTIME_REPORTING_June_2023.csv', low_memory=False, usecols=cols)
    bts_l = pd.read_csv('BTS_ONTIME_REPORTING_July_2023.csv', low_memory=False, usecols=cols)
    bts = pd.concat([bts_j, bts_l], ignore_index=True)
    del bts_j, bts_l
    bts = bts.rename(columns={'OP_UNIQUE_CARRIER': 'OP_CARRIER'})
    print(f"Full BTS loaded: {len(bts):,} rows")

    bts_dep_pool, bts_arr_pool = compute_all_traffic_metrics(bts)

    update_filtered_file('input_data_medium_filtered.csv', bts_dep_pool, bts_arr_pool)
    update_filtered_file('input_data_small_filtered.csv', bts_dep_pool, bts_arr_pool)

if __name__ == '__main__':
    main()
