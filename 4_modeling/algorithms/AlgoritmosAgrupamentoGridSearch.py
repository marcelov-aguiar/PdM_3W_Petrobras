# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:00:07 2024

@author: ADM
"""

import numpy as np
import pandas as pd
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
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import warnings

# Custom imports
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import util
from class_manipulates_path import ManipulatePath
from class_preprocessing_refactor import Preprocessing
from class_format_data import FormatData

warnings.filterwarnings("ignore")
util.init()

def split_dataframe(df, target_column=None, test_size=0.0003, random_state=42):
    if target_column:
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in split.split(df, df[target_column]):
            df_1 = df.iloc[train_index]
            df_2 = df.iloc[test_index]
    else:
        df_1, df_2 = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_1, df_2

# Load custom Data Set
manipulate_path = ManipulatePath()
preprocessing = Preprocessing()
format_data = FormatData()

df_preprocessing = pd.read_parquet(manipulate_path.get_path_preprocessing_real_data_all_classes())
df_1, df_2 = split_dataframe(df_preprocessing, target_column='class')

# aplicando PCA
n_components = 2
pca = PCA(n_components=n_components)
scaler = StandardScaler()
X = scaler.fit_transform(df_2.drop(columns="class"))
X_pca = pca.fit_transform(X)
list_comp = [f"COMP_{x+1}"  for x in range(n_components)]
df_pca = pd.DataFrame(X_pca, columns=list_comp)
df_pca["class"] = df_2["class"].values
df_2 = df_pca.copy()

# Carregamento dos dados
X = df_2.iloc[:, :-1].values

y = df_2.iloc[:, -1].values



# Declare an empty DataFrame for results
pred_df = pd.DataFrame(columns=["Algorithm", "Params", "Metric", "Score"])

def hyperparameter_search(X, param_grids):
    best_results = []
    
    for model_name, param_grid in param_grids.items():
        best_score = -1
        best_params = None
        print(f"Testando hiperparâmetros para {model_name}.")
        for params in ParameterGrid(param_grid):
            if model_name == "DBSCAN":
                model = DBSCAN(**params)
            elif model_name == "K-Means":
                model = KMeans(**params)
            elif model_name == "Hierarchical":
                model = AgglomerativeClustering(**params)
            elif model_name == "Gaussian Mixture":
                model = GaussianMixture(**params)
            elif model_name == "Mean Shift":
                model = MeanShift(**params)
            
            labels = model.fit_predict(X)
            
            # Calcula o Silhouette Score apenas se houver mais de um cluster
            if len(np.unique(labels)) > 1 and np.max(labels) < 10:
                score = silhouette_score(X, labels)
                
                # Armazena a melhor combinação para o modelo atual
                if score > best_score:
                    best_score = score
                    best_params = params
        
        # Salva o melhor resultado para o modelo atual
        if best_params:
            best_results.append({"Algorithm": model_name, "Best Params": best_params, "Silhouette Score": best_score})
    
    return pd.DataFrame(best_results)

# Definição dos grids de parâmetros
param_grids = {
    "DBSCAN": {"eps": [0.5, 0.7, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 15, 20,40], "min_samples": [1, 3, 5, 7, 9]},
    "K-Means": {"n_clusters": [2, 3, 4], "random_state": [42]},
    "Hierarchical": {"n_clusters": [2, 3, 4], "linkage": ["ward", "complete", "average"]},
    "Gaussian Mixture": {"n_components": [2, 3, 4], "covariance_type": ["full", "tied"]},
    "Mean Shift": {"bandwidth": [None, 1, 2]},
}

# execução da busca de hiperparâmetros
pred_df = hyperparameter_search(X, param_grids)

# Exibe os melhores resultados para cada modelo
print(pred_df)