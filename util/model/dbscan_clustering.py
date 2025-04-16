from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
import constants.columns_dataframe as const


class DBSCANClustering(BaseEstimator, TransformerMixin):

    def __init__(self,
                 eps: float = 0.05,
                 min_samples: int = 4,
                 apply_grid_search: bool=False,
                 eps_values: List[float] = [0.01, 0.1, 0.5, 1.0],
                 min_samples_values: List[int] = [2, 5, 10]):
        self.eps = eps
        self.min_samples = min_samples
        self.apply_grid_search = apply_grid_search
        self.eps_values = eps_values
        self.min_samples_values = min_samples_values
        

    def fit(self, X:pd.DataFrame=None, y=None):
        if not self.apply_grid_search:
            self.dbscan = DBSCAN(eps=self.eps,
                                 min_samples=self.min_samples,
                                 n_jobs=-1)
        else:
            best_eps, best_min_samples = self.__grid_search_dbscan(X,
                                                                  eps_values=self.eps_values,
                                                                  min_samples_values=self.min_samples_values)
            self.dbscan = DBSCAN(eps=best_eps,
                                 min_samples=best_min_samples,
                                 n_jobs=-1)
        self.dbscan.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[const.PRED] = pd.Series(self.dbscan.labels_)

        return X
    
    def __grid_search_dbscan(self,
                             X: pd.DataFrame,
                             eps_values: float,
                             min_samples_values: int) -> Tuple[float, int]:
        """
        Realiza uma busca em grade para encontrar os melhores valores de eps e min_samples para o DBSCAN.

        Parâmetros:
            - X: Matriz de features.
            - eps_values: Lista de valores para o parâmetro eps.
            - min_samples_values: Lista de valores para o parâmetro min_samples.

        Retorna:
            - Melhores valores de eps e min_samples.
        """
        best_eps = None
        best_min_samples = None
        best_score = np.inf

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                labels = dbscan.fit_predict(X)
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:  # Verifica se há mais de um cluster
                    score = davies_bouldin_score(X, labels)
                    if score < best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples

        return best_eps, best_min_samples


if __name__ == "__main__":
    print("End")
