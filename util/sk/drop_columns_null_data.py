from typing import List
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class DropNullData(BaseEstimator, TransformerMixin):
    def __init__(self,
                 columns_to_null_data: List[str] = None,
                 rate: float = 0.1):
        self.rate = rate
        self.columns_to_null_data = columns_to_null_data

    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        null_sensor:List[str] = []

        if self.columns_to_null_data is None:
            self.columns_to_null_data = X.columns

        for column in self.columns_to_null_data:
            if self.__has_null_data(X, column):
                null_sensor.append(column)
        
        X = X.drop(columns=null_sensor)
        return X
    
    def __has_null_data(self,
                        X: pd.DataFrame,
                        column: str) -> bool:
        percent = (X[column].isna().sum() / X.shape[0])
        if percent > self.rate:
            return 1
        else:
            return 0