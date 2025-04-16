# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:00:07 2024

@author: ADM
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import (
    DBSCAN,
    KMeans,
    AgglomerativeClustering,
    MeanShift,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    rand_score,
    fowlkes_mallows_score,
)

# custom
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import util
from class_manipulates_path import ManipulatePath
from class_preprocessing_refactor import Preprocessing
from class_format_data import FormatData
util.init()


def split_dataframe(df, target_column=None, test_size=0.0003, random_state=42):
    if target_column:
        # Se a coluna alvo for fornecida, realizar a divisão estratificada
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in split.split(df, df[target_column]):
            df_1 = df.iloc[train_index]
            df_2 = df.iloc[test_index]
    else:
        # Se a coluna alvo não for fornecida, realizar a divisão simples
        df_1, df_2 = train_test_split(df, test_size=test_size, random_state=random_state)

    return df_1, df_2

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Load custom Data Set
manipulate_path = ManipulatePath()

preprocessing = Preprocessing()

format_data = FormatData()

path_raw_data = manipulate_path.get_path_raw_data()

df_preprocessing = pd.read_parquet(manipulate_path.get_path_preprocessing_real_data_all_classes())

df_1, df_2 = split_dataframe(df_preprocessing, target_column='class')

X = df_2.iloc[:, :-1].values

y = df_2.iloc[:, -1].values

# Declare an empty DataFrame with the required columns
pred_df = pd.DataFrame(columns=["Algorithm", "Metric", "Score"])

# Implement clustering algorithms
dbscan = DBSCAN(eps=0.5, min_samples=5)
kmeans = KMeans(n_clusters=3, random_state=42)
agglo = AgglomerativeClustering(n_clusters=3)
gmm = GaussianMixture(n_components=3, covariance_type="full")
ms = MeanShift()

# Evaluate clustering algorithms with three evaluation metrics
labels = {
    "DBSCAN": dbscan.fit_predict(X),
    "K-Means": kmeans.fit_predict(X),
    "Hierarchical": agglo.fit_predict(X),
    "Gaussian Mixture": gmm.fit_predict(X),
    "Mean Shift": ms.fit_predict(X),
}

metrics = {
    "Silhouette Score": silhouette_score,
    "Calinski Harabasz Score": calinski_harabasz_score,
    "Davies Bouldin Score": davies_bouldin_score,
    "Rand Score": rand_score,
    "Fowlkes-Mallows Score": fowlkes_mallows_score,
}

results = []

for name, label in labels.items():
    for metric_name, metric_func in metrics.items():
        if metric_name in ["Rand Score", "Fowlkes-Mallows Score"]:
            score = metric_func(y, label)
        else:
            if len(np.unique(label)) != 1:
                score = metric_func(X, label)
            else:
                score = -1
        results.append({"Algorithm": name, "Metric": metric_name, "Score": score})

pred_df = pd.concat([pred_df, pd.DataFrame(results)], ignore_index=True)

# Display the DataFrame
print(pred_df.head(10))