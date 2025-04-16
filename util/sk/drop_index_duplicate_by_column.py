from typing import List
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class DropIndexDuplicateByColumn(BaseEstimator, TransformerMixin):
    def __init__(self,
                 column: str):
        self.column = column

    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        duplicated_indices = X.index[X.index.duplicated(keep=False)]
        df_duplicated = X.loc[duplicated_indices]

        # Verificar se há valores diferentes na coluna "class" para os índices duplicados
        different_class = df_duplicated.groupby(level=0).filter(lambda x: x[self.column].nunique() > 1)

        # Remover os registros identificados do DataFrame original
        df_cleaned = X.drop(different_class.index)
        return df_cleaned
