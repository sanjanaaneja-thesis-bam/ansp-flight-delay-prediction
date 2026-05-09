import time, calendar
from datetime import datetime
from datetime import time as datetime_time
import networkx as nx
import pandas as pd
import numpy as np
import math
from sympy import *


def generate_delay_score_csv_file(df, airport_list, date_range, my_alpha, my_beta, my_gamma,
                                   delay_threshold, hours_to_fly_threshold, outputname):
    df_delay_score = pd.DataFrame(columns=['Airport', 'Initial_z_score', 'Delay_Score', 'Datetime'])
    for date1 in date_range:
        my_date = date1.date()
        for time_hour in range(0, 24):
            z = find_delay_vector(df, my_gamma, airport_list, delay_threshold, time_hour, my_date)
            W = find_Matrix_W(df, my_beta, airport_list, hours_to_fly_threshold, time_hour, my_date)
            converged_delay_score = propagation(W, z, my_alpha)
            df_tmp = converged_delay_score.reset_index()
            df_tmp.columns = ['Airport', 'Delay_Score']
            df_tmp['Initial_z_score'] = df_tmp['Airport'].map(z)
            time_component = datetime_time(time_hour, 0)
            combined_datetime = datetime.combine(my_date, time_component)
            df_tmp['Datetime'] = combined_datetime
            df_delay_score = pd.concat([df_delay_score, df_tmp], ignore_index=True)
    df_delay_score.to_csv(outputname, index=False)
    return df_delay_score


def find_delay_vector(df, my_gamma, airport_list, delay_threshold, time_hour, my_date):
    reference_datetime = pd.to_datetime(my_date) + pd.Timedelta(hours=time_hour)
    filtered_df = df.loc[
        (df['ARR_DEL15'] == 1) &
        (df['Actual_ARR_dt_EST'] <= reference_datetime) &
        (df['Actual_ARR_dt_EST'] >= reference_datetime - pd.Timedelta(hours=delay_threshold))
    ]
    filtered_df['elapsed_hours'] = reference_datetime - filtered_df['Actual_ARR_dt_EST']
    filtered_df['elapsed_hours'] = filtered_df['elapsed_hours'].apply(
        lambda x: math.ceil(x.total_seconds() / 3600))

    z = {key: 0 for key in airport_list}
    for airport in airport_list:
        tmp_df = filtered_df.loc[filtered_df['DEST'] == airport]
        if not tmp_df.empty:
            z[airport] = np.exp(-my_gamma * tmp_df['elapsed_hours']).sum()
    return z


def find_Matrix_W(df, my_beta, airport_list, hours_to_fly_threshold, time_hour, my_date):
    reference_datetime = pd.to_datetime(my_date) + pd.Timedelta(hours=time_hour)
    filtered_df = df.loc[
        (df['Scheduled_ARR_EST'] >= reference_datetime) &
        (df['Scheduled_ARR_EST'] <= reference_datetime + pd.Timedelta(hours=hours_to_fly_threshold))
    ]
    # BUG FIX: original used unfiltered df instead of filtered_df
    filtered_df['to_fly_hours'] = filtered_df['Scheduled_ARR_EST'] - reference_datetime
    filtered_df['to_fly_hours'] = filtered_df['to_fly_hours'].apply(
        lambda x: math.ceil(x.total_seconds() / 3600))

    W = {key1: {key2: 0 for key2 in airport_list} for key1 in airport_list}
    num_nodes = len(airport_list)
    for idx1 in range(num_nodes):
        A = airport_list[idx1]
        for idx2 in range(num_nodes):
            B = airport_list[idx2]
            df_A_B = filtered_df.loc[(filtered_df['ORIGIN'] == A) & (filtered_df['DEST'] == B)]
            if not df_A_B.empty:
                # BUG FIX: original used filtered_df (global min) instead of df_A_B (route-specific min)
                min_to_fly_hours = df_A_B['to_fly_hours'].min()
                W[A][B] = np.exp(-my_beta * min_to_fly_hours)

    matrix = np.array([list(inner_dict.values()) for inner_dict in W.values()])
    column_sums = np.sum(matrix, axis=0)
    normalized_matrix = matrix / column_sums

    for i, (key, inner_dict) in enumerate(W.items()):
        W[key] = {inner_key: normalized_matrix[i, j] for j, inner_key in enumerate(inner_dict.keys())}
    return W


def propagation(W, z, alpha):
    df_W = pd.DataFrame(W).fillna(0)
    df_z = pd.Series(z)

    epsilon = 0.001
    max_iteration = 10000
    iter_k = 0
    diff = 10

    prior_ksi = df_z.copy()
    current_ksi = df_z.copy()
    while iter_k < max_iteration and diff > epsilon:
        iter_k += 1
        current_ksi = alpha * np.dot(df_W, prior_ksi) + (1 - alpha) * df_z
        vector_diff = current_ksi - prior_ksi
        diff = abs(max(vector_diff, key=abs))
        prior_ksi = current_ksi.copy()
    return current_ksi
