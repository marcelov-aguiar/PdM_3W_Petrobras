import pandas as pd
import logging
import numpy as np
from typing import List
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
    def remove_duplicates_index(df_data: pd.DataFrame) -> pd.DataFrame:
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
        logger.info("============ Remove duplicates DataFrame ============")
        size_duplicate_data = df_data.index.duplicated().sum()
        if size_duplicate_data != 0:
            length = df_data.shape[0]
            df_data = df_data.index.drop_duplicates()
            logger.info(f"{(size_duplicate_data/length)*100:.2f}% of the data is duplicated.")
            logger.info(f"Were removed {size_duplicate_data} duplicate data.")
        else:
            logger.info(f"No duplicate index was found.")
        return df_data
    
    @staticmethod
    def check_null_dataframe(df_data: pd.DataFrame) -> pd.DataFrame:
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
        logger.info("============ Check null DataFrame ============")
        length = df_data.shape[0]
        for column in df_data.columns:
            percent = (df_data[column].isna().sum() / length) * 100
            if percent != 0:
                logger.info(f"{percent:.2f}% are null for {column}.")
            else:
                logger.info(f"No data null was found for {column}.")

    @staticmethod
    def check_negative_number(df_data: pd.DataFrame) -> pd.DataFrame:
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
        logger.info("============ Check negative number ============")
        for tag_name in df_data.columns:
            amount_negative_number = 0
            amount_negative_number = df_data[df_data[tag_name] < 0].shape[0]
            if (amount_negative_number != 0) and (not (amount_negative_number is None)):
                logger.info(f"Amount of negative numbers for the {tag_name} are {amount_negative_number}")
            else:
                logger.info(f"No negative number was found for {tag_name}.")

    @staticmethod
    def remove_low_variance(df_data: pd.DataFrame,
                            min_variance: int = 0.05) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        df_data : pd.DataFrame
            _description_
        min_variance : int, optional
            _description_, by default 0.1

        Returns
        -------
        pd.DataFrame
            _description_
        """
        bad_sensor:List[str] = []
        for sensor in df_data.columns:
            if df_data[sensor].var() <= min_variance or (df_data[sensor].mean() == 0):
                logger.info(f"Sensor is flat: {sensor}")
                bad_sensor.append(sensor)

        if len(bad_sensor) != 0:
            logger.info(f"Feature with low variance are {bad_sensor}")
            df_data = df_data.drop(bad_sensor, axis=1)
            logger.info(f"Amouts features removed are {len(bad_sensor)}")
        else:
            logger.info("No data with low variance was found.")
        return df_data
    
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
    def resample_dataframe(df_data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        df_data : pd.DataFrame
            _description_
        frequency : str
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        return df_data.resample(frequency).mean()
    
    @staticmethod
    def remove_outlier_3_sigma(df_data: pd.DataFrame,
                               tag_name: str) -> pd.Series:
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
        means = df_data[tag_name].mean()
        stds = df_data[tag_name].std()

        # Define the lower and upper limit to identify outliers
        lower_limit = means - 3 * stds
        upper_limit = means + 3 * stds

        # Replace the null values outside the limits with NaN
        df_data.loc[(df_data[tag_name] < lower_limit) | (df_data[tag_name] > upper_limit), tag_name] = np.nan
        return df_data[tag_name]

    @staticmethod
    def iqr_score_outlier_detection(data: pd.Series,
                                    window_size: float,
                                    threshold: float) -> np.array:
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
        outliers = []

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

        return outliers