from typing import List
import pandas as pd
from sk.base_transform import BaseTransform


class Preprocessor():
    def __init__(self, transformers: List[BaseTransform]) -> None:
        self.transformers = transformers

    def transform(self, df_data: pd.DataFrame) -> pd.DataFrame:
        for transformer in self.transformers:
            df_data = transformer.transform(df_data)
        return df_data
 