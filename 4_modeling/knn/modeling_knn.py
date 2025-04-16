import util
from pathlib import Path
from typing import List
import pandas as pd
import constants.columns_dataframe as const
from experiment_one import ExperimentOne

class ModelingKNN:
    def __init__(self) -> None:
        pass

    def run_modeling(self,
                     df_preprocessing: pd.DataFrame,
                     csv_path: Path,
                     target: str):

        df_metric_1 = ExperimentOne().transform(df_preprocessing)
        print("teste")

if __name__ == "__main__":
    pass