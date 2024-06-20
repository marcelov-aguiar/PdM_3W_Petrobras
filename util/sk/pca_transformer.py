from typing import List
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA

class PCATransform(BaseEstimator, TransformerMixin):
    def __init__(self,
                 columns_to_transform: List[str],
                 n_components: int):
        self.columns_to_transform = columns_to_transform
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X:pd.DataFrame=None, y=None):
        self.pca.fit(X[self.columns_to_transform])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_data = X.copy()
        data = self.pca.transform(df_data[self.columns_to_transform])
        df_data = self.__formats_transformed_dataframe(data, X)

        return df_data
    
    def __formats_transformed_dataframe(self,
                                        data: np.array,
                                        X: pd.DataFrame) -> pd.DataFrame:
        list_comp = [f"COMP_{x+1}"  for x in range(self.n_components)]
        df_data = pd.DataFrame(data, columns=list_comp, index=X.index)
        df_data = pd.concat([df_data, X.drop(columns=self.columns_to_transform)], axis=1)
        return df_data


if __name__ == "__main__":
    # caso de teste
    dados = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(dados)

    
    pca = PCATransform(columns_to_transform=['feature1', 'feature2'],
                       n_components=1)

    
    dados = pca.fit_transform(df)
    print("End")
