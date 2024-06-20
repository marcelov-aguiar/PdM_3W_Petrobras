from typing import List
import pandas as pd
import constants.columns_dataframe as const
from sk.drop_columns import DropColumns
from sk.drop_columns_null_data import DropNullData
from sk.drop_columns_low_variance import DropLowVariance
from sk.drop_index_duplicate import DropIndexDuplicate
from sk.drop_nan import DropNaN
from sklearn.base import TransformerMixin, BaseEstimator


class PreprocessingDBSCAN():
    def __init__(self) -> None:
        pass
    def transform(self, df_data: pd.DataFrame) -> pd.DataFrame:
        df_data = DropColumns(const.BAD_COLUMNS).transform(df_data)
        df_data = DropNullData(df_data.columns).transform(df_data)
        df_data = DropLowVariance(df_data.drop(columns=[const.TARGET])).transform(df_data)
        df_data = DropIndexDuplicate().transform(df_data)
        df_data = DropNaN().transform(df_data)
        return df_data

        
class Preprocessor():
    def __init__(self, transformers: List[DropNullData]) -> None:
        self.transformers = transformers
    
    def transform(self, df_data: pd.DataFrame) -> pd.DataFrame:
        for transformer in self.transformers:
            df_data = transformer.transform(df_data)
        return df_data
      
### Sugestão de melhoria:
# transformers = [
#     DropColumns(const.BAD_COLUMNS),
#     DropNullData(df_data.columns),
#     DropLowVariance(threshold=0.1),
#     DropIndexDuplicate(),
#     DropNaN()
# ]
# 
# preprocessor = Preprocessor(transformers)
# preprocessed_data = preprocessor.transform(df_data)