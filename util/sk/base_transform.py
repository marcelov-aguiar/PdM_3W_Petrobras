from typing import List
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from abc import ABC, abstractmethod

class BaseTransform(BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, X:pd.DataFrame=None, y=None):
        pass

    abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
