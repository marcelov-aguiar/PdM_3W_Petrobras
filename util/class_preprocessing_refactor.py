import pandas as pd
import logging
import numpy as np
from typing import List, Dict
from pathlib import Path
from class_format_data import FormatData

# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# endregion

class Preprocessing:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def read_dataset_time_series(path: Path,
                                 index_name: str) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        path : Path
            _description_
        index_name : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        df_data = pd.read_csv(path)
        df_data = \
                FormatData.set_index_dataframe(df_data=df_data,
                                               column_name=index_name)
        return df_data

    @staticmethod
    def counter_class(df_data: pd.DataFrame,
                      columns_class_name: str,
                      anomaly: int) -> int:
        amount = (df_data[columns_class_name] == anomaly).sum()
    
        return amount

    @staticmethod
    def check_duplicates_index(df_data: pd.DataFrame) -> int:
        """Remove registros duplicados no DataFrame.

        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame com possíveis dados duplicados.

        Returns
        -------
        pd.DataFrame
            DataFrame sem dados duplicados
        """
        size_duplicate_data = df_data.index.duplicated().sum()
        return size_duplicate_data
    
    @staticmethod
    def amount_register(df_data: pd.DataFrame) -> int:
        return df_data.shape[0]
    
    @staticmethod
    def amount_columns(df_data: pd.DataFrame) -> int:
        return df_data.shape[1]

    @staticmethod
    def is_categorical_columns(df_data: pd.Series) -> int:
        if len(df_data.unique()) < 10:
            return 1
        else:
            return 0
        

    @staticmethod
    def check_null_dataframe(df_data: pd.DataFrame,
                             column_name: str) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        df_data : pd.DataFrame
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        #logger.info("============ Check null DataFrame ============")
        length = df_data.shape[0]
        percent = (df_data[column_name].isna().sum() / length) * 100
        return percent


    @staticmethod
    def check_negative_number(df_data: pd.DataFrame,
                              column_name: str) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        df_data : pd.DataFrame
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        if Preprocessing.is_categorical_columns(df_data[column_name]):
            return "Categorical"
        amount_negative_number = 0
        amount_negative_number = df_data[df_data[column_name] < 0].shape[0]
        return amount_negative_number

    @staticmethod
    def check_column_low_variance(df_data: pd.DataFrame,
                                  column_name: str,
                                  min_variance: int = 0.05) -> bool:
        """_summary_

        Parameters
        ----------
        df_data : pd.DataFrame
            _description_
        min_variance : int, optional
            _description_, by default 0.1

        Returns
        -------
        bool
            _description_
        """
        if df_data[column_name].var() <= min_variance or (df_data[column_name].mean() == 0):
            return 1
        else:
            return 0
    
    @staticmethod
    def remove_null_data(df_data: pd.DataFrame, rate: float= 0.1) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        df_data : pd.DataFrame
            _description_
        rate : float, optional
            Porcentagem de dados nulos, by default 0.1

        Returns
        -------
        pd.DataFrame
            _description_
        """
        logger.info("============ Remove null DataFrame ============")
        length = df_data.shape[0]
        null_sensor:List[str] = []
        for column in df_data.columns:
            percent = (df_data[column].isna().sum() / length)
            if percent > rate:
                percent = percent*100
                null_sensor.append(column)
                logger.info(f"{column} with {percent:.2f}% of null data. It was removed.")
        
        df_data = df_data.drop(columns=null_sensor)
        return df_data

    @staticmethod
    def check_amount_outlier_3_sigma(df_data: pd.DataFrame,
                                     column_name: str) -> int:
        """Remove outliers using the 3-sigma method.

        Parameters
        ----------
        df_data : pd.DataFrame
            Dataframe with the data to be analyzed.
        tag_name : str
            Name of the tag for which outliers will be removed.

        Returns
        -------
        pd.DataFrame
            Returns the dataframe of tag_name without outliers.
        """
        # Calculate the mean and standard deviation for each column
        means = df_data[column_name].mean()
        stds = df_data[column_name].std()

        # Define the lower and upper limit to identify outliers
        lower_limit = means - 3 * stds
        upper_limit = means + 3 * stds

        # Replace the null values outside the limits with NaN
        amount_data = df_data.shape[0]
        amount_outliers = len(df_data.loc[(df_data[column_name] < lower_limit) | (df_data[column_name] > upper_limit), column_name])
        return (amount_outliers/amount_data)*100

    @staticmethod
    def check_amount_iqr_score_outlier(df_data: pd.DataFrame,
                                       column_name: str,
                                       window_size: float,
                                       threshold: float) -> int:
        """IQR Score method to detect local outliers

        Parameters
        ----------
        data : pd.Series
            Numpy array containing your original data
        window_size : float
            Local window size
        threshold : float
            Limit for detection of outliers

        Returns
        -------
        np.array
            Array containing the indices of the data points that
            are identified as outliers
        """
        if Preprocessing.is_categorical_columns(df_data[column_name]):
            return "Categorical"

        outliers = []
        data = df_data[column_name].copy()
        # Loop through the data
        for i in range(len(data)):
            # Define the start and end indices for the local window
            start_idx = max(0, i - window_size)
            end_idx = min(len(data) - 1, i + window_size)

            # Calculate the IQR for the local window
            local_data = data[start_idx:end_idx+1]
            q1 = np.percentile(local_data, 25)
            q3 = np.percentile(local_data, 75)
            iqr = q3 - q1

            # Define the upper and lower thresholds
            upper_threshold = q3 + threshold * iqr
            lower_threshold = q1 - threshold * iqr

            # Check if the data point is an outlier
            if data[i] > upper_threshold or data[i] < lower_threshold:
                outliers.append(i)

        return len(outliers)