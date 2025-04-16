import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging
from class_manipulates_path import ManipulatePath

# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# endregion


class FormatData(ManipulatePath):
    """É reponsável por formatar os dados brutos para o formato
    de DataFrame para ser possível fazer a análise exploratória.

    Parameters
    ----------
    ManipulatePath : object
        Guardar informações da localização dos
        repositórios e dos dados específicos do projeto.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def read_data(csv_path: Path,
                  index_name: str) -> pd.DataFrame:
        """Responsável por fazer a leitura dos dados

        Parameters
        ----------
        csv_path : Path
            Path do arquivo
        index_name : str

        Returns
        -------
        pd.DataFrame
            DataFrame com os dados brutos.
        """
        df_data = pd.read_csv(str(csv_path))
        df_data = FormatData.set_index_dataframe(df_data=df_data,
                                                 column_name=index_name)
        return df_data

    def format_raw_data(self,
                        input_file_name: str) -> None:
        """Responsável por fazer a leitura do dado bruto no formato
        .txt, renomear as colunas para os nomes definidos em
        `get_features_name` e salvar o arquivo em formato csv.

        Parameters
        ----------
        input_file_name : Path
            Nome do arquivo que contém os dados do equipamento que
            será lido os dados. Pode assumir os seguintes valores:
            X_FD00Y.txt. Onde X pode assumir valor `test` ou `train`.
            E Y pode assumir os valores 1, 2, 3 ou 4.
        """
        path_raw_data = self.get_path_raw_data()

        path_data = path_raw_data.joinpath(input_file_name)

        data = np.loadtxt(path_data)

        features_name = self.get_features_name()

        df_data = pd.DataFrame(data, columns=features_name)

        get_path_exploratory_data = self.get_path_exploratory_data()

        input_file_name = os.path.splitext(input_file_name)[0]
        output_file_name = f"{input_file_name}_format.parquet"

        path_output_format_data = \
            get_path_exploratory_data.joinpath(output_file_name)

        df_data.to_parquet(path_output_format_data, index=False)

    @staticmethod
    def set_index_dataframe(df_data: pd.DataFrame,
                            column_name: str) -> pd.DataFrame:
        """Define índice do DataFrame como `column_name` e faz a
        ordenação.

        Raise
        -----
        `column_name` deve ser do tipo Timestamp.

        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame com dados de `column_name` definidos.
        column_name : str
            Nome da coluna a ser um índice do DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame com índice definido.
        """
        df_data = df_data.set_index(column_name)
        try:
            df_data.index = pd.to_datetime(df_data.index)
            df_data = df_data.sort_index()
        except Exception as e:
            logger.error(f"Não foi possível converter {column_name} para timestamp.")
            logger.error(e)
        return df_data

    def get_format_data(self,
                        output_file_name: str) -> pd.DataFrame:
        """Responsável por fazer a leitura do arquivo salvo por
        `format_raw_data`.

        Parameters
        ----------
        output_file_name : str
            Nome do arquivo do equipamento que será feito a leitura.

        Returns
        -------
        pd.DataFrame
            DataFrame dos dados do equipamento.
        """
        get_path_exploratory_data = self.get_path_exploratory_data()

        output_file_name = os.path.splitext(output_file_name)[0]
        try:
            df_data = pd.read_parquet(
                os.path.join(get_path_exploratory_data,
                             f"{output_file_name}_format.parquet"))

        except Exception as e:
            logger.error("Dado não encontrado. Execute método " +
                         f"format_raw_data para {output_file_name}.")
            logger.error(e)

        return df_data
