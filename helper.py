import time, calendar
from datetime import datetime
from datetime import time as datetime_time
import networkx as nx
import pandas as pd
import numpy as np
import math
from sympy import *


def generate_month_date_range(year, month):
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}'
    return pd.date_range(start=start_date, end=end_date, freq='D')


def calculate_lambda(df):
    # λ such that w_{AB} ≈ 1 for the route with the maximum (elapsed_time / distance) ratio
    x = symbols('x')
    df['ratio'] = df['ACTUAL_ELAPSED_TIME'] / df['DISTANCE']
    largest_ratio = df['ratio'].max()
    expression = exp(largest_ratio * x) / (1 + exp(largest_ratio * x))
    my_lambda = solve(expression - 0.9999, x)
    return float(my_lambda[0])


def data_preprocess(df, variables_selected):
    new_df = df[variables_selected].dropna().copy()
    new_df['Scheduled_ARR_EST'] = pd.to_datetime(new_df['Scheduled_ARR_EST'])
    new_df['Actual_ARR_dt_EST'] = pd.to_datetime(new_df['Actual_ARR_dt_EST'])
    new_df['MONTH'] = new_df['MONTH'].astype(int)
    new_df['ARR_DEL15'] = new_df['ARR_DEL15'].astype(int)
    new_df['Actual_ARR_Hour'] = new_df['Actual_ARR_dt_EST'].dt.hour
    return new_df
