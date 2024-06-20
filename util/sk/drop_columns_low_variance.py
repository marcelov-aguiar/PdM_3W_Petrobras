from typing import List
import pandas as pd
from sk.base_transform import BaseTransform

class DropLowVariance(BaseTransform):
    def __init__(self,
                 exception_columns: List[str]= None,
                 min_variance: float = 0.01):
        self.exception_columns = exception_columns
        self.min_variance = min_variance

    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        low_variance_attr:List[str] = []

        columns_check: List[str] = list(X.columns)
        
        if self.exception_columns is not None:
            for column in self.exception_columns:
                if column in columns_check:
                    columns_check.remove(column)

            
        for column in columns_check:
            if self.__has_column_low_variance(X, column):
                low_variance_attr.append(column)
        
        X = X.drop(columns=low_variance_attr)
        return X
    
    def __has_column_low_variance(self,
                                  df_data: pd.DataFrame,
                                  column_name: str) -> bool:
        if df_data[column_name].var() <= self.min_variance or (df_data[column_name].mean() == 0):
            return 1
        else:
            return 0
