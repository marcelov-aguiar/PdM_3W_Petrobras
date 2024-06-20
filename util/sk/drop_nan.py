from typing import List
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class DropNaN(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.dropna()
        return X
