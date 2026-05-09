#RQ2 parameter grid search: 27 (alpha, beta, gamma) combinations evaluated with XGBoost OOT split

import os
import sys
import calendar
import warnings
import itertools
import time as time_module
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import helper, delay_score

# CONFIGURATION

INPUT_FILE = 'input_data_unified_60.csv'
FAA_FILE   = 'FAA_selected_airports_sample.csv'

# RQ1 folders (read-only, not written to by this script)
RQ1_NETWORK_DIR = 'thesis_outputs/network_feature'

# RQ2 folders
RQ2_BASE        = 'thesis_outputs/rq2'
RQ2_DELAY_DIR   = 'thesis_outputs/rq2/delay_score'
RQ2_RESULTS_DIR = 'thesis_outputs/rq2/results'
RQ2_HEATMAP_DIR = 'thesis_outputs/rq2/results/heatmaps'

# Fixed thresholds (same as authors and RQ1)
START_MONTH           = 6
END_MONTH             = 7
DELAY_THRESHOLD       = 2
HOURS_TO_FLY          = 5

# Parameter grid
ALPHA_VALS = [0.75, 0.85, 0.95]
BETA_VALS  = [1.5,  2.3,  3.5]
GAMMA_VALS = [0.7,  0.9,  0.99]

ALL_COMBOS = list(itertools.product(ALPHA_VALS, BETA_VALS, GAMMA_VALS))  # 27 combos

# STEP 1. COMPUTE ANSP DELAY SCORES 

def delay_score_path(month, alpha, beta, gamma):
    #Return the RQ2 delay score file path for a given parameter combo.
    return os.path.join(
        RQ2_DELAY_DIR,
        f'month={month}alpha={alpha}beta={beta}gamma={gamma}delay_score.csv'
    )

def ensure_delay_scores(df_processed, alpha, beta, gamma):
    """
    Compute and cache ANSP delay scores for both months if not already on disk.
    Mirrors the delay score computation in thesis_main.py.
    """
    for month in [START_MONTH, END_MONTH]:
        out_path = delay_score_path(month, alpha, beta, gamma)
        if os.path.exists(out_path):
            print(f"  Cached: {os.path.basename(out_path)}")
            continue

        print(f"  Computing: month={month} alpha={alpha} beta={beta} gamma={gamma}",
              flush=True)
        t0 = time_module.time()

        df_month   = df_processed[df_processed['MONTH'] == month]
        date_range = helper.generate_month_date_range(2023, month)

        delay_score.generate_delay_score_csv_file(
            df_month, selected_airport_list, date_range,
            alpha, beta, gamma,
            DELAY_THRESHOLD, HOURS_TO_FLY,
            out_path
        )
        elapsed = time_module.time() - t0
        print(f"  Done ({elapsed/60:.1f} min): {os.path.basename(out_path)}")

# STEP 2. LOAD AND MERGE DATA FOR PREDICTION

def load_base_data():
    print("Loading unified dataset")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = df[df['MONTH'].isin([START_MONTH, END_MONTH])].copy()
    faa = pd.read_csv(FAA_FILE)
    airports = sorted(faa['Airport_Code'].tolist())
    df = df[df['ORIGIN'].isin(airports) & df['DEST'].isin(airports)].copy()
    print(f"  Rows after filter: {len(df):,}")

    # Preserve raw ORIGIN before one-hot encoding (needed for ANSP merge in merge_ansp)
    # Named with leading underscore so it doesn't match startswith('ORIGIN_') feature selector
    df['_origin_key'] = df['ORIGIN'].copy()

    # Truncate departure time to hour for ANSP merge
    df['Scheduled_DEP_EST_adj'] = pd.to_datetime(df['Scheduled_DEP_EST']).dt.floor('h')

    # Load network centrality features (same files as RQ1, unchanged)
    net_dfs = []
    for m in [START_MONTH, END_MONTH]:
        path = os.path.join(RQ1_NETWORK_DIR, f'network_feature_month={m}.csv')
        nf = pd.read_csv(path)
        nf = nf.set_index(nf.columns[0]).T
        nf.reset_index(inplace=True)
        nf.rename(columns={'index': 'Airport'}, inplace=True)
        nf['month'] = m
        net_dfs.append(nf)
    net_var = pd.concat(net_dfs, axis=0)

    df = df.merge(net_var, left_on=["ORIGIN", "MONTH"],
                  right_on=["Airport", "month"], how='left')
    df = df.drop(columns=[c for c in df.columns if c == 'Airport'], errors='ignore')

    # Rename to match RQ1 convention
    df = df.rename(columns={
        "Arr_1hrpre_num":                  "Arr_1hr_Prior_Num",
        "Arr_1hrpost_num":                 "Arr_1hr_Post_Num",
        "DEP_1hrpre_num":                  "Dep_1hr_Prior_Num",
        "DEP_1hrpost_num":                 "Dep_1hr_Post_Num",
        "max_temp_f":                      "Max_Temp_F",
        "min_temp_f":                      "Min_Temp_F",
        "avg_wind_speed_kts":              "Avg_Wind_Speed_Kts",
        "precip_in":                       "Precip_In",
        "affected_turnaround_lessthan60":  "Scheduled_Turnaround_Lessthan60",
        "betweenness_centrality":          "Betweenness_Centrality",
        "closeness_centrality":            "Closeness_Centrality",
    })

    # One-hot encode categoricals (full 60-airport set for consistency)
    Carrier    = pd.get_dummies(df[["OP_CARRIER"]]).iloc[:, :-1]
    day_of_week = pd.get_dummies(df[["day_of_week"]]).iloc[:, :-1]
    ORIGIN     = pd.get_dummies(df[["ORIGIN"]]).iloc[:, :-1]
    df = pd.concat([df, Carrier, day_of_week, ORIGIN], axis=1)
    df = df.drop(['OP_CARRIER', 'day_of_week', 'ORIGIN'], axis=1)

    # Define feature sets (no ANSP columns yet, added per combo)
    variables_baseline = (
        ['Arr_1hr_Prior_Num', 'Arr_1hr_Post_Num', 'Dep_1hr_Prior_Num', 'Dep_1hr_Post_Num',
         'Max_Temp_F', 'Min_Temp_F', 'Avg_Wind_Speed_Kts', 'Precip_In',
         'Scheduled_Turnaround_Lessthan60']
        + [c for c in df.columns if c.startswith('OP_CARRIER_')]
        + [c for c in df.columns if c.startswith('day_of_week_')]
        + [c for c in df.columns if c.startswith('ORIGIN_')]
    )
    variables_centrality = ['Betweenness_Centrality', 'Closeness_Centrality']

    # Drop rows with NaN in baseline/centrality features to match RQ1's dropna scope
    # (RQ1 runs dropna across all model columns including weather and traffic counts)
    na_check_cols = [
        'Arr_1hr_Prior_Num', 'Arr_1hr_Post_Num', 'Dep_1hr_Prior_Num', 'Dep_1hr_Post_Num',
        'Max_Temp_F', 'Min_Temp_F', 'Avg_Wind_Speed_Kts', 'Precip_In',
        'Scheduled_Turnaround_Lessthan60', 'Betweenness_Centrality', 'Closeness_Centrality',
    ]
    before = len(df)
    df = df.dropna(subset=[c for c in na_check_cols if c in df.columns])
    print(f"  Dropped {before - len(df):,} rows with NaN in baseline/centrality features ({len(df):,} remain)")

    feature_meta = {
        'variables_baseline':   variables_baseline,
        'variables_centrality': variables_centrality,
        'airports':             airports,
    }

    return df, feature_meta

def merge_ansp(df_base, alpha, beta, gamma):
    """
    Merge RQ2 ANSP delay scores into df_base for the given combo.
    Merges at 2hr, 3hr, and 6hr lags to match RQ1's dropna scope (all-lag
    row filter), ensuring RQ1 and RQ2 train/test on identical flight rows.
    Only 2hr score is used as a model feature. 3hr/6hr are dropped after filtering.
    """
    ds_dfs = []
    for m in [START_MONTH, END_MONTH]:
        path = delay_score_path(m, alpha, beta, gamma)
        ds = pd.read_csv(path)
        ds_dfs.append(ds)
    delay_scores = pd.concat(ds_dfs, axis=0)
    delay_scores['Datetime']       = pd.to_datetime(delay_scores['Datetime'])
    delay_scores['post2_Datetime'] = delay_scores['Datetime'] + timedelta(hours=2)
    delay_scores['post3_Datetime'] = delay_scores['Datetime'] + timedelta(hours=3)
    delay_scores['post6_Datetime'] = delay_scores['Datetime'] + timedelta(hours=6)

    df = df_base.copy()

    # Merge at 2hr (feature), 3hr and 6hr (row-filter parity with RQ1 only)
    for post_col, score_col in [
        ('post2_Datetime', 'ANSP_Score_rq2'),
        ('post3_Datetime', '_ansp_3hr_filter'),
        ('post6_Datetime', '_ansp_6hr_filter'),
    ]:
        df = df.merge(
            delay_scores[["Airport", "Delay_Score", post_col]],
            left_on=["_origin_key", "Scheduled_DEP_EST_adj"],
            right_on=["Airport", post_col],
            how='left'
        )
        df = df.rename(columns={"Delay_Score": score_col})
        df = df.drop(columns=[c for c in df.columns if c == 'Airport'], errors='ignore')

    # Dropna matching RQ1's all-lag filter so both scripts train on identical rows
    model_cols = ['DEP_DEL15', 'ANSP_Score_rq2', '_ansp_3hr_filter', '_ansp_6hr_filter',
                  'Betweenness_Centrality', 'Closeness_Centrality', 'FAA_class', 'MONTH']
    reg_data = df.dropna(subset=[c for c in model_cols if c in df.columns])

    # Drop the filter-only lag columns before modelling
    reg_data = reg_data.drop(columns=['_ansp_3hr_filter', '_ansp_6hr_filter'], errors='ignore')
    return reg_data

# STEP 3. XGBOOST OOT EVALUATION

def run_xgboost_oot(reg_data, feature_meta, hub, alpha, beta, gamma):
    #Train XGBoost on June, evaluate on July for a single hub and parameter combo.
    
    vb = feature_meta['variables_baseline']
    vc = feature_meta['variables_centrality']

    if hub == 'All':
        data = reg_data.copy()
    else:
        data = reg_data[reg_data['FAA_class'] == hub].copy()

    if len(data) < 100:
        print(f"    WARNING: only {len(data)} rows for {hub}, skipping")
        return None

    train = data[data['MONTH'] == START_MONTH].copy()
    test  = data[data['MONTH'] == END_MONTH].copy()

    if len(train) < 50 or len(test) < 50:
        print(f"    WARNING: insufficient train/test rows for {hub}, skipping")
        return None

    baseline_cols = [c for c in vb + vc if c in data.columns]
    ansp_cols     = baseline_cols + ['ANSP_Score_rq2']

    results = {}

    for label, cols in [('Baseline', baseline_cols), ('ANSP_2hr', ansp_cols)]:
        available = [c for c in cols if c in data.columns]
        X_train = train[available]
        y_train = train['DEP_DEL15'].values.ravel()
        X_test  = test[available]
        y_test  = test['DEP_DEL15'].values.ravel()

        spw = 1 / np.mean(y_train) - 1
        clf = XGBClassifier(
            n_estimators=200, random_state=0,
            scale_pos_weight=spw,
            verbosity=0, use_label_encoder=False, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, preds)
        results[label] = round(auc, 6)

    delta = results['ANSP_2hr'] - results['Baseline']
    print(f"    {hub:8s}  Baseline={results['Baseline']:.4f}  "
          f"ANSP_2hr={results['ANSP_2hr']:.4f}  DAUC={delta:+.4f}",
          flush=True)

    return {
        'Alpha': alpha, 'Beta': beta, 'Gamma': gamma,
        'Hub': hub,
        'N_train': len(train), 'N_test': len(test),
        'AUC_Baseline': results['Baseline'],
        'AUC_ANSP_2hr': results['ANSP_2hr'],
        'DAUC': round(delta, 6),
    }

# STEP 4. HEATMAP VISUALISATION

def plot_heatmaps(df_results):
    os.makedirs(RQ2_HEATMAP_DIR, exist_ok=True)

    for hub in ['Large', 'Medium', 'Small', 'All']:
        for gamma in GAMMA_VALS:
            subset = df_results[
                (df_results['Hub'] == hub) &
                (df_results['Gamma'] == gamma)
            ]
            if subset.empty:
                continue

            pivot = subset.pivot(index='Beta', columns='Alpha', values='DAUC')

            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                pivot, annot=True, fmt='.4f', cmap='YlOrRd',
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'DAUC (ANSP 2hr vs Baseline)'}
            )
            ax.set_title(
                f'RQ2 DAUC Heatmap: {hub} Hub, gamma={gamma}',
                fontsize=12
            )
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Beta')
            plt.tight_layout()

            fname = f'rq2_heatmap_{hub.lower()}_gamma{str(gamma).replace(".", "")}.png'
            out_path = os.path.join(RQ2_HEATMAP_DIR, fname)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved heatmap: {fname}")

# MAIN

def main():
    os.makedirs(RQ2_DELAY_DIR,   exist_ok=True)
    os.makedirs(RQ2_RESULTS_DIR, exist_ok=True)

    global selected_airport_list
    faa = pd.read_csv(FAA_FILE)
    selected_airport_list = sorted(faa['Airport_Code'].tolist())
    print(f"Airport network: {len(selected_airport_list)} airports")

    df_all = pd.read_csv(INPUT_FILE, low_memory=False)
    df_all = df_all[
        (df_all['ORIGIN'].isin(selected_airport_list)) &
        (df_all['DEST'].isin(selected_airport_list))
    ].copy()
    variables_selected = ['MONTH', 'ORIGIN', 'DEST', 'ARR_DEL15', 'DISTANCE',
                          'Scheduled_ARR_EST', 'Actual_ARR_dt_EST']
    df_processed = helper.data_preprocess(df_all, variables_selected)
    print(f"  Preprocessed: {len(df_processed):,} rows")

    df_base, feature_meta = load_base_data()

    # Phase 1: ensure all 27 x 2 delay score files exist
    total_combos = len(ALL_COMBOS)
    for i, (alpha, beta, gamma) in enumerate(ALL_COMBOS, 1):
        print(f"\nCombo {i}/{total_combos}: alpha={alpha} beta={beta} gamma={gamma}")
        ensure_delay_scores(df_processed, alpha, beta, gamma)

    # Phase 2: run XGBoost OOT for all combos

    all_results = []
    intermediate_csv = os.path.join(RQ2_RESULTS_DIR, 'rq2_results.csv')

    for i, (alpha, beta, gamma) in enumerate(ALL_COMBOS, 1):
        print(f"\nCombo {i}/{total_combos}: alpha={alpha} beta={beta} gamma={gamma}")

        reg_data = merge_ansp(df_base, alpha, beta, gamma)
        print(f"  Rows after ANSP merge: {len(reg_data):,}")

        for hub in ['Large', 'Medium', 'Small', 'All']:
            result = run_xgboost_oot(reg_data, feature_meta, hub, alpha, beta, gamma)
            if result:
                all_results.append(result)

        # Save intermediate results after each combo
        pd.DataFrame(all_results).to_csv(intermediate_csv, index=False)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(intermediate_csv, index=False)
    print(f"\nResults saved to {intermediate_csv}")

    print("\nBest combo per hub (DAUC):")
    for hub in ['Large', 'Medium', 'Small', 'All']:
        h = df_results[df_results['Hub'] == hub]
        if h.empty:
            continue
        best = h.loc[h['DAUC'].idxmax()]
        print(f"  {hub:8s}  best DAUC={best['DAUC']:+.4f}  "
              f"alpha={best['Alpha']} beta={best['Beta']} gamma={best['Gamma']}")

    print("\nAuthors default combo (alpha=0.85, beta=2.3, gamma=0.9):")
    default = df_results[
        (df_results['Alpha'] == 0.85) &
        (df_results['Beta']  == 2.3)  &
        (df_results['Gamma'] == 0.9)
    ]
    for _, row in default.iterrows():
        print(f"  {row['Hub']:8s}  DAUC={row['DAUC']:+.4f}  "
              f"AUC_ANSP={row['AUC_ANSP_2hr']:.4f}")

    # Phase 3: heatmaps
    plot_heatmaps(df_results)
    print(f"\nResults: {intermediate_csv}")
    print(f"Heatmaps: {RQ2_HEATMAP_DIR}")

if __name__ == '__main__':
    main()
