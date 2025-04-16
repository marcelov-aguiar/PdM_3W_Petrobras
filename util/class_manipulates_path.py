from pathlib import Path
from typing import List

path_manipulate_data = Path(__file__).parent


class ManipulateFileName():
    def __init__(self) -> None:
        pass

    def get_file_name_report_raw_data(self) -> str:
        return "report_raw_data.xlsx"

    def get_file_name_data_all_classes(self) -> str:
        return "preprocessing_real_data_all_classes.parquet"
    
    def get_file_name_draw_data(self) -> str:
        return "preprocessing_draw_data.parquet"
    
    def get_file_name_real_data_two_classes(self) -> str:
        return "preprocessing_real_data_two_classes.parquet"

    def get_file_name_simulated_data(self) -> str:
        return "preprocessing_simulated_data.parquet"

class ManipulatePath(ManipulateFileName):
    """Responsável por guardar informações da localização dos
    repositórios e dos dados específicos do projeto.
    """
    def __init__(self) -> None:
        pass

    def get_path_data_all(self) -> Path:
        """Path com todos os dados:
        - Dados brutos
        - Dados formatados na análise exploratória
        - Dados pré-processados

        Returns
        -------
        Path
            Path com todos os dados.
        """
        path_raw_data = path_manipulate_data.parent.joinpath("1_data")

        return path_raw_data

    def get_path_raw_data(self) -> Path:
        """Responsável por retornar o path onde fica
        localizado os dados brutos dos equipamentos do
        projeto.

        Returns
        -------
        Path
            path onde fica localizado os dados brutos dos
            equipamentos do projeto.
        """
        path_raw_data = path_manipulate_data.parent.joinpath(self.get_path_data_all(),
                                                             '1_raw_data')

        return path_raw_data
    
    def get_path_report_raw_data(self) -> Path:
        path_report_raw_data = path_manipulate_data.parent.joinpath(self.get_path_data_all(),
                                                                    '2_report_raw_data',
                                                                    self.get_file_name_report_raw_data())

        return path_report_raw_data

    def get_path_exploratory_data(self) -> Path:
        """Responsável por retornar o path onde fica
        localizados os dados analisados na etapa de
        análise exploratória.

        Returns
        -------
        Path
            path onde fica localizados os dados analisados
            na etapa de análise exploratória.
        """
        path_exploratory_data = \
            path_manipulate_data.parent.joinpath(self.get_path_data_all(),
                                                 '2_formatted_data')

        return path_exploratory_data

    def get_path_preprocessing_output(self) -> Path:
        """Responsável por retornar o path onde fica localizado
        os dados de saída do pré-processamento.

        Returns
        -------
        Path
            path onde fica localizado os dados de saída do
            pré-processamento.
        """
        path_preprocessing_output = \
            path_manipulate_data.parent.joinpath(self.get_path_data_all(),
                                                 '3_preprocessed_data')

        return path_preprocessing_output

    def get_path_preprocessing_real_data_all_classes(self):
        preprocessing_real_data_all_classes = \
            path_manipulate_data.parent.joinpath(self.get_path_data_all(),
                                                 '3_preprocessing',
                                                 self.get_file_name_data_all_classes())

        return preprocessing_real_data_all_classes
    
    def get_path_preprocessing_draw_data(self):
        path_preprocessing_draw_data = \
            path_manipulate_data.parent.joinpath(self.get_path_data_all(),
                                                 '3_preprocessing',
                                                 self.get_file_name_draw_data())

        return path_preprocessing_draw_data
    

    def get_path_preprocessing_real_data_two_classes(self):
        preprocessing_real_data_two_classes = \
            path_manipulate_data.parent.joinpath(self.get_path_data_all(),
                                                 '3_preprocessing',
                                                 self.get_file_name_real_data_two_classes())

        return preprocessing_real_data_two_classes
    
    def get_path_preprocessing_simulated_data(self):
        path_preprocessing_simulated_data = \
            path_manipulate_data.parent.joinpath(self.get_path_data_all(),
                                                 '3_preprocessing',
                                                 self.get_file_name_simulated_data())

        return path_preprocessing_simulated_data
    
    def get_path_img(self) -> Path:
        """Reponsável por retornar o caminho onde estão salvas as imagens
        importantes do projeto

        Returns
        -------
            Path
                Caminho onde se localiza as imagens usada no texto final do projeto.
        """
        return path_manipulate_data.parent.joinpath("img")

    def get_path_mlflow(self) -> Path:
        """Reponsável por retornar o caminho onde estão salvos os artefatos
        do MLFlow.

        Returns
        -------
        Path
            Caminho com os artefatos do MLFlow.
        """
        return path_manipulate_data.parent.joinpath("testeMLFlow", "mlruns")

if __name__ == "__main__":
    manipulate_data = ManipulatePath()
