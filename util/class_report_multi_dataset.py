"Classe responsável por fazer report de vários datasets."
import pandas as pd
from pathlib import Path
from typing import List
import logging
from class_manipulates_path import ManipulatePath
from class_preprocessing import Preprocessing
from class_format_data import FormatData
from class_preprocessing_refactor import Preprocessing


# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# endregion

class ManipulaExcelFile():
    def __init__(self) -> None:
        self.column_file_name = 'File name'
        self.column_feature_name = 'Feature name'
        self.column_null_data_name = 'Percent null data'
        self.column_negative_number_name = 'Percent negative number'
        self.column_low_variance_name = 'Low variance'
        self.column_outlier_name = 'Percent outliers IQR'
        self.columns_amount_register_name = "Amount Register"
        self.columns_amount_columns_name = "Amount Columns"
        self.columns_duplicate_index_name = "Duplicates Index"
        self.dict_class = {
            0: "Without anomaly",
            1: "Anomaly 1",
            101: "Anomaly 1",
            2: "Anomaly 2",
            102: "Anomaly 2",
            3: "Anomaly 3",
            103: "Anomaly 3",
            4: "Anomaly 4",
            104: "Anomaly 4",
            5: "Anomaly 5",
            105: "Anomaly 5",
            6: "Anomaly 6",
            106: "Anomaly 6",
            7: "Anomaly 7",
            107: "Anomaly 7",
            8: "Anomaly 8",
            108: "Anomaly 8",

        }

    def create_excel_variables(self) -> pd.DataFrame:
        df_variable = pd.DataFrame(columns=[self.column_file_name,
                                            self.column_feature_name,
                                            self.column_null_data_name,
                                            self.column_negative_number_name,
                                            self.column_low_variance_name,
                                            self.column_outlier_name
                                            ])
        return df_variable

class ReportMultiDataset(ManipulaExcelFile):
    def __init__(self, output_path: Path) -> None:
        self.output_path = str(output_path)
        super().__init__()


    def process_all_csv(self, list_path: List[Path]):
        df_output = pd.DataFrame()
        counter = len(list_path)
        i = 1
        for path in list_path:
            logger.info(f'{i} arquivos de {counter}')
            path = str(path)
            df_data = Preprocessing.read_dataset_time_series(path,
                                                             "timestamp")
            file_name = "\\".join(path.split("\\")[-2:])
            df_staticts = self.process_df(df_data, file_name)
            df_output = pd.concat([df_output, df_staticts])
            i = i + 1
        df_output.to_excel(self.output_path, index=False)

    def process_df(self,
                   df_data: pd.DataFrame,
                   file_name: str) -> pd.DataFrame:
        df_variable = pd.DataFrame()
        for column in df_data.columns:
            df_aux = pd.DataFrame()

            percent_null_data = Preprocessing.check_null_dataframe(df_data=df_data,
                                                                   column_name=column)

            amount_negative_number = Preprocessing.check_negative_number(df_data=df_data,
                                                                         column_name=column)

            is_low_variance = Preprocessing.check_column_low_variance(df_data=df_data,
                                                                      column_name=column)

            amount_outlier = Preprocessing.check_amount_outlier_3_sigma(df_data=df_data,
                                                                        column_name=column)
            df_aux[self.column_file_name] = pd.Series(file_name)
            df_aux[self.column_feature_name] = pd.Series(column)
            df_aux[self.column_null_data_name] = pd.Series(percent_null_data)
            df_aux[self.column_negative_number_name] = pd.Series(amount_negative_number)
            df_aux[self.column_low_variance_name] = pd.Series(is_low_variance)
            df_aux[self.column_outlier_name] = pd.Series(amount_outlier)

            df_variable = pd.concat([df_variable, df_aux])

        
        df_variable[self.column_file_name] = file_name

        df_variable[self.columns_amount_register_name] = \
            Preprocessing.amount_register(df_data=df_data)

        df_variable[self.columns_amount_columns_name] = \
            Preprocessing.amount_columns(df_data=df_data)

        df_variable[self.columns_duplicate_index_name] = \
            Preprocessing.check_duplicates_index(df_data=df_data)
        
        for anomaly in self.dict_class.keys():
            amount_class = Preprocessing.counter_class(df_data=df_data,
                                                       columns_class_name="class",
                                                       anomaly=anomaly)
            df_variable[f'Amount {anomaly}'] = amount_class

        return df_variable

if __name__ == "__main__":
    manipulate_path = ManipulatePath()
    path_raw_data = manipulate_path.get_path_raw_data()
    #file_name = "WELL-00001_20140124093303.csv"
    list_file_path = list(path_raw_data.rglob("*.csv"))
    
    #file_path = str(path_raw_data.joinpath("1").joinpath(file_name))
    #df_data = pd.read_csv(file_path)

    #df_data = FormatData.set_index_dataframe(df_data=df_data,
    #                                         column_name="timestamp")
    report = ReportMultiDataset(str(manipulate_path.get_path_report_raw_data()))
    report.process_all_csv(list_file_path)
    # report.process_df(df_data=df_data,
    #                   file_name=file_name)
    print("Final teste")