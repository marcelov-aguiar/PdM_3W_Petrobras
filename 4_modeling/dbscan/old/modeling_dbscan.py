from pathlib import Path
import pandas as pd
from dbscan.preprocessing_dbscan import PreprocessingDBSCAN
import constants.columns_dataframe as const
from experiment_one import ExperimentOne


class ModelingDBSCAN:
    def __init__(self) -> None:
        pass

    def run_modeling(self,
                     df_raw_data: pd.DataFrame,
                     csv_path: Path,
                     target: str):
        df_preprocessing = PreprocessingDBSCAN().transform(df_raw_data)
        #TODO: Fazer report do dataset

        df_metric_1 = ExperimentOne().transform(df_preprocessing)
        print("teste")

if __name__ == "__main__":
    pass