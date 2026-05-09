#Compute ANSP delay scores and network features for the 60-airport network.

import os
import sys
import time
import calendar
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sympy import *
import networkx as nx
import pandas as pd

import helper, delay_score, network_feature

INPUT_FILE = 'input_data_unified_60.csv'
FAA_FILE = 'FAA_selected_airports_sample.csv'
OUTPUT_BASE = 'thesis_outputs'

# ANSP hyperparameters (Tan et al. defaults)
start_month = 6
end_month = 7
alpha = 0.85
beta = 2.3
gamma = 0.9
delay_threshold = 2
hours_to_fly_threshold = 5

faa = pd.read_csv(FAA_FILE)
selected_airport_list = sorted(faa['Airport_Code'].tolist())
print(f"Airport network: {len(selected_airport_list)} airports "
      f"(Large={len(faa[faa['Hub_Category']=='Large'])}, "
      f"Medium={len(faa[faa['Hub_Category']=='Medium'])}, "
      f"Small={len(faa[faa['Hub_Category']=='Small'])})")

df_all_data = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Loaded {INPUT_FILE}: {len(df_all_data):,} rows")

df_selected = df_all_data[
    (df_all_data['ORIGIN'].isin(selected_airport_list)) &
    (df_all_data['DEST'].isin(selected_airport_list))
].copy()
print(f"  After ORIGIN+DEST filter: {len(df_selected):,} rows")

variables_selected = ['MONTH', 'ORIGIN', 'DEST', 'ARR_DEL15', 'DISTANCE',
                      'Scheduled_ARR_EST', 'Actual_ARR_dt_EST']
df_processed = helper.data_preprocess(df_selected, variables_selected)
print(f"  After preprocess: {len(df_processed):,} rows")

output_delay_score = os.path.join(OUTPUT_BASE, 'delay_score')
output_network_feature = os.path.join(OUTPUT_BASE, 'network_feature')
output_figure = os.path.join(OUTPUT_BASE, 'figure')
for d in [output_delay_score, output_network_feature, output_figure]:
    os.makedirs(d, exist_ok=True)

print(f"\nParameters: alpha={alpha}, beta={beta}, gamma={gamma}, "
      f"delay_threshold={delay_threshold}h, hours_to_fly={hours_to_fly_threshold}h")

for month in range(start_month, end_month + 1):
    print(f"\nMonth {month}: {len(df_processed[df_processed['MONTH']==month]):,} flights")
    df_month = df_processed[df_processed['MONTH'] == month]

    t0 = time.time()
    network_feature_filename = os.path.join(
        output_network_feature, f'network_feature_month={month}.csv')
    frequency_df = network_feature.create_frequency_matrix(
        df_month, month, selected_airport_list)
    network_feature.get_graph_features(
        frequency_df, network_feature_filename).T.rename_axis('Airport').reset_index()
    print(f"  Network features: {network_feature_filename} ({time.time()-t0:.1f}s)")

    csv_file_name = os.path.join(
        output_delay_score,
        f'month={month}alpha={alpha}beta={beta}gamma={gamma}delay_score.csv')
    t0 = time.time()
    df_delay_score = delay_score.generate_delay_score_csv_file(
        df_month, selected_airport_list, helper.generate_month_date_range(2023, month),
        alpha, beta, gamma, delay_threshold, hours_to_fly_threshold,
        csv_file_name)
    print(f"  Delay scores: {csv_file_name} ({time.time()-t0:.1f}s)")

    df_delay_score['Datetime'] = pd.to_datetime(df_delay_score['Datetime'])
    df_delay_score['Hour'] = df_delay_score['Datetime'].dt.hour
    average_delay_hourly = df_delay_score.groupby('Hour')['Delay_Score'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(average_delay_hourly, marker='o', linestyle='-')
    plt.title(f'month={month} alpha={alpha} beta={beta} gamma={gamma} Average Delay Score')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Delay Score')
    plt.xticks(range(24))
    plt.grid(True)
    plt.tight_layout()
    figure_path = os.path.join(
        output_figure,
        f'month={month}alpha={alpha}beta={beta}gamma={gamma}_avg_delay_score.pdf')
    plt.savefig(figure_path)
    plt.close()
    print(f"  Figure: {figure_path}")
