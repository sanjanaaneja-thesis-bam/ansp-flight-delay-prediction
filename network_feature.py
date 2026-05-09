import time, calendar
from datetime import datetime
from datetime import time as datetime_time
import networkx as nx
import pandas as pd
import numpy as np
import math
from sympy import *


def create_frequency_matrix(df, month, top_airports):
    df_filtered = df[df['MONTH'] == month]
    frequency_matrix = pd.crosstab(df_filtered['ORIGIN'], df_filtered['DEST'],
                                    rownames=['ORIGIN'], colnames=['DEST'], dropna=False)
    frequency_df = frequency_matrix.reindex(index=top_airports, columns=top_airports, fill_value=0)
    return pd.DataFrame(frequency_df)


def get_graph_features(df, outputname):
    column_names = list(df.columns)
    G = nx.DiGraph()
    num_nodes = len(column_names)
    for idx1 in range(num_nodes):
        node1 = column_names[idx1]
        for idx2 in range(num_nodes):
            if idx1 == idx2:
                continue
            node2 = column_names[idx2]
            edge_weight = df[column_names[idx1]].iloc[idx2]
            if edge_weight > 0:
                G.add_edge(node1, node2, weight=edge_weight)

    graph_features = {}
    for node in G.nodes():
        graph_features[node] = {
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node),
            'weighted_in_degree': G.in_degree(weight='weight')[node],
            'weighted_out_degree': G.out_degree(weight='weight')[node],
            'betweenness_centrality': nx.betweenness_centrality(G)[node],
            'closeness_centrality': nx.closeness_centrality(G)[node],
            'in_degree_centrality': nx.in_degree_centrality(G)[node],
            'out_degree_centrality': nx.out_degree_centrality(G)[node],
        }

    new_df = pd.DataFrame(graph_features)
    new_df.to_csv(outputname)
    return new_df
