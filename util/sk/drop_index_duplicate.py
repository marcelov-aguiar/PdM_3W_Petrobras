from typing import List
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class DropIndexDuplicate(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.__has_duplicates_index(X):
            X = X[~X.index.duplicated()]
        return X

    def __has_duplicates_index(self,
                               X: pd.DataFrame) -> int:
        """Remove registros duplicados no DataFrame.

        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame com possíveis dados duplicados.

        Returns
        -------
        pd.DataFrame
            DataFrame sem dados duplicados
        """
        size_duplicate_data = X.index.duplicated().sum()
        if size_duplicate_data > 0:
            return 1
        else:
            return 0
