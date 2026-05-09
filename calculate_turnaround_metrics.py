#Compute aircraft turnaround metrics from the full BTS dataset for all 3 hub files.

import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

if sys.platform == 'win32':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

def hhmm_to_minutes(hhmm_series):
    #Convert HHMM integer series to minutes-since-midnight. NaN for invalid.
    h = pd.to_numeric(hhmm_series, errors='coerce')
    hours = h // 100
    minutes = h % 100
    mask = (h.notna()) & (hours < 24) & (minutes < 60)
    result = pd.Series(np.nan, index=h.index)
    result[mask] = (hours[mask] * 60 + minutes[mask]).astype(float)
    return result

def build_local_datetimes(df):
    #Build local dep/arr datetime columns from BTS HHMM fields (no timezone conversion).
    df = df.copy()
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')

    crs_dep_min = hhmm_to_minutes(df['CRS_DEP_TIME'])
    dep_min = hhmm_to_minutes(df['DEP_TIME'])
    crs_arr_min = hhmm_to_minutes(df['CRS_ARR_TIME'])
    arr_min = hhmm_to_minutes(df['ARR_TIME'])

    base_date = df['FL_DATE']

    # Scheduled departure: always on FL_DATE
    df['_dep_local_sched'] = base_date + pd.to_timedelta(crs_dep_min, unit='m')

    # Scheduled arrival: +1 day if CRS_ARR < CRS_DEP (overnight)
    sched_arr_offset = (crs_arr_min < crs_dep_min).fillna(False).astype(int)
    df['_arr_local_sched'] = (base_date
                              + pd.to_timedelta(sched_arr_offset, unit='D')
                              + pd.to_timedelta(crs_arr_min, unit='m'))

    # Actual departure: +1 day if scheduled late evening (>=1800) and actual early morning (<600)
    dep_day_offset = ((crs_dep_min >= 1080) & (dep_min < 360)).fillna(False).astype(int)
    df['_dep_local_actual'] = (base_date
                               + pd.to_timedelta(dep_day_offset, unit='D')
                               + pd.to_timedelta(dep_min, unit='m'))

    # Actual arrival: inherits dep_day_offset, +1 more if ARR < DEP (actual times)
    arr_extra_offset = (arr_min < dep_min).fillna(False).astype(int)
    arr_total_offset = dep_day_offset + arr_extra_offset
    df['_arr_local_actual'] = (base_date
                               + pd.to_timedelta(arr_total_offset, unit='D')
                               + pd.to_timedelta(arr_min, unit='m'))

    return df

def compute_turnaround_full_bts(bts):
  # Compute turnaround for every flight where prev.DEST == curr.ORIGIN (same aircraft).
    bts = build_local_datetimes(bts)

    # Exclude cancelled/diverted so shift(1) finds the last operated previous leg
    n_before = len(bts)
    operated = bts['DEP_TIME'].notna() & bts['ARR_TIME'].notna()
    bts = bts[operated].copy()
    print(f"Filtered to operated flights (dropped {n_before - len(bts):,} cancelled/diverted)")

    bts = bts.sort_values(['TAIL_NUM', '_dep_local_sched'], na_position='last').reset_index(drop=True)

    # Drop rows with missing TAIL_NUM (can't attribute to an aircraft)
    n_before = len(bts)
    bts = bts[bts['TAIL_NUM'].notna()].reset_index(drop=True)
    print(f"    Dropped {n_before - len(bts):,} rows with missing TAIL_NUM")

    # Initialize turnaround columns
    bts['scheduled_Turnarnd'] = np.nan
    bts['Actual_Turnarnd'] = np.nan

    # Vectorized approach: compute prev columns via groupby.shift
    bts['_prev_DEST'] = bts.groupby('TAIL_NUM', sort=False)['DEST'].shift(1)
    bts['_prev_arr_sched'] = bts.groupby('TAIL_NUM', sort=False)['_arr_local_sched'].shift(1)
    bts['_prev_arr_actual'] = bts.groupby('TAIL_NUM', sort=False)['_arr_local_actual'].shift(1)

    # Valid turnaround rows: prev.DEST == curr.ORIGIN (aircraft arrived at this airport)
    valid = (bts['_prev_DEST'] == bts['ORIGIN']) & bts['_prev_DEST'].notna()
    print(f"    Valid turnaround rows: {valid.sum():,}/{len(bts):,} ({valid.mean()*100:.1f}%)")

    # Scheduled turnaround = curr_sched_dep - prev_sched_arr (same airport, same TZ)
    sched_valid = valid & bts['_dep_local_sched'].notna() & bts['_prev_arr_sched'].notna()
    bts.loc[sched_valid, 'scheduled_Turnarnd'] = (
        (bts.loc[sched_valid, '_dep_local_sched'] - bts.loc[sched_valid, '_prev_arr_sched'])
        .dt.total_seconds() / 60
    )

    # Actual turnaround = curr_actual_dep - prev_actual_arr
    actual_valid = valid & bts['_dep_local_actual'].notna() & bts['_prev_arr_actual'].notna()
    bts.loc[actual_valid, 'Actual_Turnarnd'] = (
        (bts.loc[actual_valid, '_dep_local_actual'] - bts.loc[actual_valid, '_prev_arr_actual'])
        .dt.total_seconds() / 60
    )

    # Data quality filter: negative turnarounds are not physically meaningful.
    # They occur when BTS has stale TAIL_NUM reassignments, carrier-level aircraft
    # swaps not reflected in the schedule, or other data entry issues. Setting
    # these to NaN prevents them from contaminating the model features (they'd
    # otherwise be flagged as "< 60 min" under our clean threshold rule).
    neg_sched = (bts['scheduled_Turnarnd'] < 0)
    n_neg_sched = int(neg_sched.sum())
    bts.loc[neg_sched, 'scheduled_Turnarnd'] = np.nan
    print(f"    Set {n_neg_sched:,} negative scheduled_Turnarnd values to NaN")

    neg_actual = (bts['Actual_Turnarnd'] < 0)
    n_neg_actual = int(neg_actual.sum())
    bts.loc[neg_actual, 'Actual_Turnarnd'] = np.nan
    print(f"    Set {n_neg_actual:,} negative Actual_Turnarnd values to NaN")

    # Diff
    bts['Diff_in_turnarnd'] = bts['Actual_Turnarnd'] - bts['scheduled_Turnarnd']

    # Binary flags (NaN if scheduled_Turnarnd is NaN)
    bts['longTurnaround'] = bts['scheduled_Turnarnd'].apply(
        lambda x: 1 if pd.notna(x) and x > 180 else (0 if pd.notna(x) else np.nan))

    # late_airjet_when_turnaround_within_180 requires LATE_AIRCRAFT_DELAY
    if 'LATE_AIRCRAFT_DELAY' in bts.columns:
        bts['late_airjet_when_turnaround_within_180'] = bts.apply(
            lambda r: 1 if (pd.notna(r['scheduled_Turnarnd']) and
                             r['scheduled_Turnarnd'] <= 180 and
                             pd.notna(r.get('LATE_AIRCRAFT_DELAY', np.nan)) and
                             r.get('LATE_AIRCRAFT_DELAY', 0) > 0)
            else (0 if pd.notna(r['scheduled_Turnarnd']) else np.nan), axis=1)
    else:
        bts['late_airjet_when_turnaround_within_180'] = np.nan

    for thresh in [120, 90, 60, 45]:
        col = f'affected_turnaround_lessthan{thresh}'
        bts[col] = bts['scheduled_Turnarnd'].apply(
            lambda x: 1 if pd.notna(x) and x < thresh else (0 if pd.notna(x) else np.nan))

    # Drop temp columns
    bts = bts.drop(columns=['_prev_DEST', '_prev_arr_sched', '_prev_arr_actual',
                             '_dep_local_sched', '_dep_local_actual',
                             '_arr_local_sched', '_arr_local_actual'])

    return bts

# MERGE BACK TO FILTERED FILES

def make_key(df, fl_date_col='FL_DATE'):
    """Composite merge key: FL_DATE|OP_CARRIER|ORIGIN|DEST|CRS_DEP_TIME."""
    fd = pd.to_datetime(df[fl_date_col], errors='coerce').dt.strftime('%Y-%m-%d')
    return (fd + '|' +
            df['OP_CARRIER'].astype(str) + '|' +
            df['ORIGIN'].astype(str) + '|' +
            df['DEST'].astype(str) + '|' +
            pd.to_numeric(df['CRS_DEP_TIME'], errors='coerce').fillna(-1).astype(int).astype(str))

def update_filtered_file(file_path, bts_turnaround):
    """Handles both Medium/Small (have CRS_DEP_TIME column) and Large"""
    df = pd.read_csv(file_path, low_memory=False)
    print(f"\nUpdating {file_path}: {len(df):,} rows")

    # Large file doesn't have CRS_DEP_TIME; extract HHMM from Scheduled_DEP
    if 'CRS_DEP_TIME' not in df.columns and 'Scheduled_DEP' in df.columns:
        sd = pd.to_datetime(df['Scheduled_DEP'], errors='coerce')
        df['CRS_DEP_TIME'] = (sd.dt.hour * 100 + sd.dt.minute).astype('Int64')
        _had_no_crs = True
    else:
        _had_no_crs = False

    df['_mkey'] = make_key(df)
    bts_turnaround = bts_turnaround.copy()
    bts_turnaround['_mkey'] = make_key(bts_turnaround)

    turn_cols = ['scheduled_Turnarnd', 'Actual_Turnarnd', 'Diff_in_turnarnd',
                 'longTurnaround', 'late_airjet_when_turnaround_within_180',
                 'affected_turnaround_lessthan120', 'affected_turnaround_lessthan90',
                 'affected_turnaround_lessthan60', 'affected_turnaround_lessthan45']
    lookup = bts_turnaround[['_mkey'] + turn_cols].drop_duplicates('_mkey', keep='first')

    # Rename target cols with _new suffix for merge
    new_cols = {c: c + '_new' for c in turn_cols}
    lookup = lookup.rename(columns=new_cols)

    df = df.merge(lookup, on='_mkey', how='left')

    # Report how many got non-NaN new turnarounds (i.e. prev.DEST == curr.ORIGIN for their aircraft)
    matched = df['scheduled_Turnarnd_new'].notna().sum()
    print(f"    Rows with valid turnaround (prev.DEST == curr.ORIGIN): {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

    # Overwrite the 9 turnaround columns
    for c in turn_cols:
        df[c] = df[c + '_new']

    # Drop temp columns (including CRS_DEP_TIME if we added it for Large)
    drop_cols = ['_mkey'] + [c + '_new' for c in turn_cols]
    if _had_no_crs and 'CRS_DEP_TIME' in df.columns:
        drop_cols.append('CRS_DEP_TIME')
    df = df.drop(columns=drop_cols)

    df.to_csv(file_path, index=False)
    new_valid = df['scheduled_Turnarnd'].notna().sum()
    print(f"  Saved {file_path}: {new_valid:,}/{len(df):,} valid turnarounds")

# MAIN

def main():
    cols = ['FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'TAIL_NUM',
            'CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME',
            'LATE_AIRCRAFT_DELAY']
    bts_j = pd.read_csv('BTS_ONTIME_REPORTING_June_2023.csv', low_memory=False, usecols=cols)
    bts_l = pd.read_csv('BTS_ONTIME_REPORTING_July_2023.csv', low_memory=False, usecols=cols)
    bts = pd.concat([bts_j, bts_l], ignore_index=True)
    del bts_j, bts_l
    bts = bts.rename(columns={'OP_UNIQUE_CARRIER': 'OP_CARRIER'})
    print(f"Full BTS loaded: {len(bts):,} rows, {bts['TAIL_NUM'].nunique():,} unique aircraft")

    bts_turn = compute_turnaround_full_bts(bts)

    # Update Medium, Small, AND Large files for consistent flag definition. simple threshold logic: flag=1 iff scheduled_Turnarnd < threshold.
    update_filtered_file('input_data_medium_filtered.csv', bts_turn)
    update_filtered_file('input_data_small_filtered.csv', bts_turn)
    update_filtered_file('input_data_large_filtered.csv', bts_turn)

if __name__ == '__main__':
    main()
