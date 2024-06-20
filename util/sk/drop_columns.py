from typing import List
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self,
                 columns_to_drop: List[str]):
        self.columns_to_drop = columns_to_drop

    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X[self.columns_to_drop]
            X = X.drop(columns=self.columns_to_drop)
        except KeyError:
            pass
        return X
