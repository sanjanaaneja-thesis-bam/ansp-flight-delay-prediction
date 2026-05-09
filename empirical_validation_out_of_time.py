#Out-of-time validation: train June 2023, test July 2023. Adapted from FlightDelayPrediction/empirical_validation.py.

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# CONFIGURATION

INPUT_FILE = 'input_data_unified_60.csv'
FAA_FILE = 'FAA_selected_airports_sample.csv'
ANSP_DIR = 'thesis_outputs'
RESULTS_DIR = 'thesis_outputs/results'

month = 6  # June = train, July (month+1) = test
alpha = 0.85
beta = 2.3
gamma = 0.9

# DATA LOADING AND MERGE  (identical to thesis_empirical_validation.py)

def load_and_merge():
    """Load unified data, merge network features and ANSP scores."""

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"{INPUT_FILE}: {len(df):,} rows")

    faa = pd.read_csv(FAA_FILE)
    all_airports = sorted(faa['Airport_Code'].tolist())

    df = df[df['MONTH'].isin([month, month + 1])].copy()
    df = df[df['ORIGIN'].isin(all_airports) & df['DEST'].isin(all_airports)].copy()
    print(f"  After ORIGIN+DEST filter: {len(df):,} rows")

    df['Scheduled_DEP_EST_adj'] = pd.to_datetime(df['Scheduled_DEP_EST']).dt.floor('h')

    print("\nLoading network features")
    net_dfs = []
    for m in [month, month + 1]:
        path = os.path.join(ANSP_DIR, 'network_feature', f'network_feature_month={m}.csv')
        nf = pd.read_csv(path)
        nf = nf.set_index(nf.columns[0]).T
        nf.reset_index(inplace=True)
        nf.rename(columns={'index': 'Airport'}, inplace=True)
        nf['month'] = m
        net_dfs.append(nf)
    net_var = pd.concat(net_dfs, axis=0)
    print(f"  Network features loaded for months {month} and {month + 1}")

    print("Loading ANSP delay scores")
    ds_dfs = []
    for m in [month, month + 1]:
        path = os.path.join(ANSP_DIR, 'delay_score',
                            f'month={m}alpha={alpha}beta={beta}gamma={gamma}delay_score.csv')
        ds = pd.read_csv(path)
        ds_dfs.append(ds)
    delay_scores = pd.concat(ds_dfs, axis=0)
    delay_scores['Datetime'] = pd.to_datetime(delay_scores['Datetime'])
    delay_scores['post2_Datetime'] = delay_scores['Datetime'] + timedelta(hours=2)
    delay_scores['post3_Datetime'] = delay_scores['Datetime'] + timedelta(hours=3)
    delay_scores['post6_Datetime'] = delay_scores['Datetime'] + timedelta(hours=6)
    print(f"  Delay scores loaded: {len(delay_scores):,} rows")

    print("\nMerging network features")
    df = df.merge(net_var, left_on=["ORIGIN", "MONTH"],
                  right_on=["Airport", "month"], how='left')

    print("Merging ANSP scores (2hr, 3hr, 6hr lags)")
    for lag, post_col, score_suffix, init_suffix in [
        (2, 'post2_Datetime', 'Delay_Score_2hr', 'Initial_z_score_2hr'),
        (3, 'post3_Datetime', 'Delay_Score_3hr', 'Initial_z_score_3hr'),
        (6, 'post6_Datetime', 'Delay_Score_6hr', 'Initial_z_score_6hr'),
    ]:
        df = df.merge(
            delay_scores[["Airport", "Delay_Score", "Initial_z_score", post_col]],
            left_on=["ORIGIN", "Scheduled_DEP_EST_adj"],
            right_on=["Airport", post_col],
            how='left'
        )
        df = df.rename(columns={
            "Delay_Score": score_suffix,
            "Initial_z_score": init_suffix,
        })
        df = df.drop(columns=[c for c in df.columns if c == 'Airport'], errors='ignore')

    df = df.rename(columns={
        "Arr_1hrpre_num": "Arr_1hr_Prior_Num",
        "Arr_1hrpost_num": "Arr_1hr_Post_Num",
        "DEP_1hrpre_num": "Dep_1hr_Prior_Num",
        "DEP_1hrpost_num": "Dep_1hr_Post_Num",
        "max_temp_f": "Max_Temp_F",
        "min_temp_f": "Min_Temp_F",
        "avg_wind_speed_kts": "Avg_Wind_Speed_Kts",
        "precip_in": "Precip_In",
        "affected_turnaround_lessthan60": "Scheduled_Turnaround_Lessthan60",
        "betweenness_centrality": "Betweenness_Centrality",
        "closeness_centrality": "Closeness_Centrality",
        "Delay_Score_2hr": "ANSP_Score",
        "Delay_Score_3hr": "ANSP_Score_3hr",
        "Delay_Score_6hr": "ANSP_Score_6hr",
    })

    print("Creating one-hot dummies (full 60-airport set)")
    Carrier = pd.get_dummies(df[["OP_CARRIER"]]).iloc[:, :-1]
    day_of_week = pd.get_dummies(df[["day_of_week"]]).iloc[:, :-1]
    ORIGIN = pd.get_dummies(df[["ORIGIN"]]).iloc[:, :-1]
    df = pd.concat([df, Carrier, day_of_week, ORIGIN], axis=1)
    df = df.drop(['OP_CARRIER', 'day_of_week', 'ORIGIN'], axis=1)

    variables_dependent = ['DEP_DEL15']

    variables_baseline = (
        ['Arr_1hr_Prior_Num', 'Arr_1hr_Post_Num', 'Dep_1hr_Prior_Num', 'Dep_1hr_Post_Num',
         'Max_Temp_F', 'Min_Temp_F', 'Avg_Wind_Speed_Kts', 'Precip_In',
         'Scheduled_Turnaround_Lessthan60']
        + [c for c in df.columns if c.startswith('OP_CARRIER_')]
        + [c for c in df.columns if c.startswith('day_of_week_')]
        + [c for c in df.columns if c.startswith('ORIGIN_')]
    )

    variables_centrality = ['Betweenness_Centrality', 'Closeness_Centrality']
    variables_delay_index = ['ANSP_Score', 'ANSP_Score_3hr', 'ANSP_Score_6hr']
    variables_delay_initial_index = ['Initial_z_score_2hr', 'Initial_z_score_3hr', 'Initial_z_score_6hr']

    flight_information = [
        "FL_DATE", "DEST", "Scheduled_DEP", "Scheduled_ARR_Local",
        "Actual_ARR_dt_Local", "Actual_DEP_dt_EST",
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
        "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
    ]

    model_cols = (variables_dependent + variables_baseline + variables_centrality
                  + variables_delay_index + variables_delay_initial_index)
    available_model = [c for c in model_cols if c in df.columns]
    missing_model = [c for c in model_cols if c not in df.columns]
    if missing_model:
        print(f"  WARNING: Missing model columns: {missing_model}")

    # Keep MONTH for the June/July OOT split
    all_keep = available_model + ['FAA_class', 'MONTH']
    for c in flight_information:
        if c in df.columns:
            all_keep.append(c)
    all_keep = list(dict.fromkeys(all_keep))

    reg_data = df[all_keep].dropna(subset=available_model)
    print(f"\n  After dropna (on model features only): {len(reg_data):,} rows (from {len(df):,})")

    june_n = (reg_data['MONTH'] == 6).sum()
    july_n = (reg_data['MONTH'] == 7).sum()
    print(f"  Train (June): {june_n:,} rows  |  Test (July): {july_n:,} rows")
    for cat in ['Large', 'Medium', 'Small']:
        n = (reg_data['FAA_class'] == cat).sum()
        print(f"    {cat}: {n:,}")
    print(f"  DEP_DEL15 rate: {reg_data['DEP_DEL15'].mean():.4f}")

    feature_sets = {
        'variables_dependent': variables_dependent,
        'variables_baseline': variables_baseline,
        'variables_centrality': variables_centrality,
        'variables_delay_index': variables_delay_index,
        'variables_delay_initial_index': variables_delay_initial_index,
        'flight_information': flight_information,
    }

    return reg_data, feature_sets

# OUT-OF-TIME MODEL TRAINING

def run_oot_models(reg_data, feature_sets, hub_category):
    """
    Out-of-time evaluation for all 4 models x 7 feature combos on a hub category.
    Train on June (MONTH==6), test on July (MONTH==7).
    Returns a list of result dicts.
    """
    vd = feature_sets['variables_dependent']
    vb = feature_sets['variables_baseline']
    vc = feature_sets['variables_centrality']
    vdi = feature_sets['variables_delay_index']
    vdii = feature_sets['variables_delay_initial_index']

    if hub_category == 'All':
        data = reg_data.copy()
    else:
        data = reg_data[reg_data['FAA_class'] == hub_category].copy()

    if len(data) < 100:
        print(f"  WARNING: Only {len(data)} rows for {hub_category}, skipping")
        return []

    train = data[data['MONTH'] == 6].copy()
    test  = data[data['MONTH'] == 7].copy()

    print(f"\n  {hub_category}: {len(data):,} total | train={len(train):,} (June) | test={len(test):,} (July)")
    print(f"  DEP_DEL15 rate  train: {train['DEP_DEL15'].mean():.4f}  test: {test['DEP_DEL15'].mean():.4f}")

    if len(train) < 100 or len(test) < 100:
        print(f"  WARNING: Insufficient train/test rows, skipping {hub_category}")
        return []

    feature_combos = {
        'Baseline+Centrality': vb + vc,
        'B+C+ANSP_2hr':  vb + vc + vdi[0:1],
        'B+C+Init_2hr':  vb + vc + vdii[0:1],
        'B+C+ANSP_3hr':  vb + vc + vdi[1:2],
        'B+C+Init_3hr':  vb + vc + vdii[1:2],
        'B+C+ANSP_6hr':  vb + vc + vdi[2:3],
        'B+C+Init_6hr':  vb + vc + vdii[2:3],
    }

    all_feature_cols = list(dict.fromkeys(vb + vc + vdi + vdii))
    available_features = [c for c in all_feature_cols if c in data.columns]

    x_train = train[available_features]
    y_train = train[vd]
    x_test  = test[available_features]
    y_test  = test[vd]

    valid_combos = {}
    for name, cols in feature_combos.items():
        valid_cols = [c for c in cols if c in data.columns]
        if len(valid_cols) == len(cols):
            valid_combos[name] = cols
        else:
            missing = set(cols) - set(valid_cols)
            print(f"  WARNING: {name} missing columns: {missing}")

    results = []

    # XGBoost
    print(f"    Running XGBoost", flush=True)
    for combo_name, combo_cols in valid_combos.items():
        spw = 1 / np.mean(y_train.values) - 1
        xg = XGBClassifier(n_estimators=200, random_state=0, scale_pos_weight=spw,
                           verbosity=0, use_label_encoder=False, n_jobs=-1)
        xg.fit(x_train[combo_cols], y_train.values.ravel())
        p = xg.predict_proba(x_test[combo_cols])[:, 1]
        preds_bin = np.round(p)
        ytrue = y_test['DEP_DEL15'].values
        auc = roc_auc_score(ytrue, p)
        f1  = f1_score(ytrue, preds_bin)
        rec = recall_score(ytrue, preds_bin)
        results.append({'Hub': hub_category, 'Model': 'XGBoost', 'Features': combo_name,
                        'AUC': round(auc, 6), 'F1': round(f1, 6),
                        'Recall': round(rec, 6), 'N_train': len(train), 'N_test': len(test)})
        print(f"      XGBoost {combo_name}: AUC={auc:.4f} F1={f1:.4f} Recall={rec:.4f}", flush=True)

    # Random Forest
    print(f"    Running Random Forest", flush=True)
    for combo_name, combo_cols in valid_combos.items():
        rf = RandomForestClassifier(n_estimators=200, random_state=0,
                                    max_features='sqrt', class_weight='balanced_subsample',
                                    n_jobs=-1)
        rf.fit(x_train[combo_cols], y_train.values.ravel())
        p = rf.predict_proba(x_test[combo_cols])[:, 1]
        preds_bin = np.round(p)
        ytrue = y_test['DEP_DEL15'].values
        auc = roc_auc_score(ytrue, p)
        f1  = f1_score(ytrue, preds_bin)
        rec = recall_score(ytrue, preds_bin)
        results.append({'Hub': hub_category, 'Model': 'RF', 'Features': combo_name,
                        'AUC': round(auc, 6), 'F1': round(f1, 6),
                        'Recall': round(rec, 6), 'N_train': len(train), 'N_test': len(test)})
        print(f"      RF {combo_name}: AUC={auc:.4f} F1={f1:.4f} Recall={rec:.4f}", flush=True)

    # Bagged Logistic Regression
    print(f"    Running Bagged Logistic Regression", flush=True)
    for combo_name, combo_cols in valid_combos.items():
        blr = BaggingClassifier(
            estimator=LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'),
            n_estimators=200, random_state=0, n_jobs=-1)
        blr.fit(x_train[combo_cols], y_train.values.ravel())
        p = blr.predict_proba(x_test[combo_cols])[:, 1]
        preds_bin = np.round(p)
        ytrue = y_test['DEP_DEL15'].values
        auc = roc_auc_score(ytrue, p)
        f1  = f1_score(ytrue, preds_bin)
        rec = recall_score(ytrue, preds_bin)
        results.append({'Hub': hub_category, 'Model': 'BaggedLR', 'Features': combo_name,
                        'AUC': round(auc, 6), 'F1': round(f1, 6),
                        'Recall': round(rec, 6), 'N_train': len(train), 'N_test': len(test)})
        print(f"      BaggedLR {combo_name}: AUC={auc:.4f} F1={f1:.4f} Recall={rec:.4f}", flush=True)

    # ANN
    # StandardScaler fit on June training data only; transform July test.
    # 20% of June carved out for early-stopping validation (not used as test).
    print(f"    Running ANN (100 epochs max, patience 5)", flush=True)
    for combo_name, combo_cols in valid_combos.items():
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=np.ravel(y_train))
        cw_dict = dict(zip(np.unique(y_train), cw))

        sc = StandardScaler()
        x_tr_sc = pd.DataFrame(sc.fit_transform(x_train), columns=available_features)
        x_te_sc = pd.DataFrame(sc.transform(x_test),  columns=available_features)

        x_tr, x_va, y_tr, y_va = train_test_split(
            x_tr_sc, y_train, test_size=0.20, random_state=0)

        n_features = len(combo_cols)
        hidden_units = int(np.round((n_features + 1) * 2 / 3))
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
        model = Sequential([
            Dense(hidden_units, activation='relu', input_shape=(n_features,)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_tr[combo_cols], y_tr,
                  validation_data=(x_va[combo_cols], y_va),
                  epochs=100, batch_size=512, verbose=0,
                  callbacks=[es], class_weight=cw_dict)

        p = model.predict(x_te_sc[combo_cols], verbose=0).flatten()
        preds_bin = np.round(p)
        ytrue = y_test['DEP_DEL15'].values
        auc = roc_auc_score(ytrue, p)
        f1  = f1_score(ytrue, preds_bin)
        rec = recall_score(ytrue, preds_bin)
        results.append({'Hub': hub_category, 'Model': 'ANN', 'Features': combo_name,
                        'AUC': round(auc, 6), 'F1': round(f1, 6),
                        'Recall': round(rec, 6), 'N_train': len(train), 'N_test': len(test)})
        print(f"      ANN {combo_name}: AUC={auc:.4f} F1={f1:.4f} Recall={rec:.4f}", flush=True)

    return results

# MAIN

def main():
    reg_data, feature_sets = load_and_merge()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []

    for hub in ['Large', 'Medium', 'Small', 'All']:
        print(f"\nHub: {hub}")
        results = run_oot_models(reg_data, feature_sets, hub)
        all_results.extend(results)

        # Save intermediate results after each hub
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(os.path.join(RESULTS_DIR, 'empirical_results_oot.csv'), index=False)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'empirical_results_oot.csv'), index=False)

    print("\nResults summary (out-of-time):")

    for hub in ['Large', 'Medium', 'Small', 'All']:
        print(f"\n{hub}")
        hub_results = df_results[df_results['Hub'] == hub]
        for model in ['XGBoost', 'RF', 'BaggedLR', 'ANN']:
            model_results = hub_results[hub_results['Model'] == model]
            if len(model_results) == 0:
                continue
            baseline = model_results[model_results['Features'] == 'Baseline+Centrality']
            ansp_2hr = model_results[model_results['Features'] == 'B+C+ANSP_2hr']
            if len(baseline) > 0 and len(ansp_2hr) > 0:
                b_auc = baseline['AUC'].values[0]
                a_auc = ansp_2hr['AUC'].values[0]
                delta = a_auc - b_auc
                print(f"  {model:12s}  Baseline AUC={b_auc:.4f}  +ANSP_2hr AUC={a_auc:.4f}  Delta={delta:+.4f}")

    print(f"\nFull results saved to {os.path.join(RESULTS_DIR, 'empirical_results_oot.csv')}")

if __name__ == '__main__':
    main()
