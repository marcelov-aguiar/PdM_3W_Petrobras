from typing import Dict
import pandas as pd
from sk.base_transform import BaseTransform


class ReplaceColumnDataFrame(BaseTransform):
    def __init__(self,
                 column_to_replace: str,
                 mapping: Dict[int, int]):
        self.column_to_replace = column_to_replace
        self.mapping = mapping

    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.__replace_values(X)
        return X
    
    def __replace_values(self,
                         df: pd.DataFrame) -> pd.DataFrame:
        for key, value in self.mapping.items():
            df[self.column_to_replace] = df[self.column_to_replace].replace(key, value)
        
        return df


if __name__ == "__main__":
    # Dicionário de mapeamento
    mapeamento = {
        "original_valor_1": "novo_valor_1",
        "original_valor_2": "novo_valor_2",
        # Adicione mais substituições conforme necessário
    }

    # DataFrame de exemplo
    df = pd.DataFrame({
        'coluna': ["original_valor_1", "original_valor_2", "outro_valor"]
    })

    df_new = ReplaceColumnDataFrame("coluna", mapeamento).transform(df)
    print("Finish")