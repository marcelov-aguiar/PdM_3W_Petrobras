import util
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sk.base_transform import BaseTransform

class NormStandardScaler(BaseTransform):
    def __init__(self,
                 columns_to_norm: List[str]):
        super().__init__()
        self.scaler = StandardScaler()
        self.columns_to_norm = columns_to_norm

    def fit(self, X:pd.DataFrame=None, y=None):
        self.scaler.fit(X[self.columns_to_norm])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_data = X.copy()
        data = self.scaler.transform(df_data[self.columns_to_norm])
        df_data = self.__formats_normalized_dataframe(data, X)

        return df_data
    
    def __formats_normalized_dataframe(self,
                                       data: np.array,
                                       X: pd.DataFrame) -> pd.DataFrame:
        df_data = pd.DataFrame(data, columns=self.columns_to_norm, index=X.index)
        df_data = pd.concat([df_data, X.drop(columns=self.columns_to_norm)], axis=1)
        return df_data


if __name__ == "__main__":
    # caso de teste
    dados = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(dados)

    
    scaler = NormStandardScaler(columns_to_norm=['feature1', 'feature2'])

    
    dados_normalizados = scaler.fit_transform(df)
    print("Finish")
