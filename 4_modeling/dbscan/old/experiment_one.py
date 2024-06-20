import pandas as pd
from sk.norm_standard_scaler import NormStandardScaler
from sk.pca_transformer import PCATransform
from model.dbscan_clustering import DBSCANClustering
from viz.visualization import plot_two_variables
import constants.columns_dataframe as const

class ExperimentOne:
    def __init__(self) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X: pd.DataFrame = NormStandardScaler(
            columns_to_norm=self.__get_norm_columns(X)
            ).fit_transform(X)

        X: pd.DataFrame = PCATransform(
            columns_to_transform=self.__get_norm_columns(X),
            n_components=const.N_COMPONENTS_PCA
            ).fit_transform(X)

        fig_real = plot_two_variables(df=X,
                                      var1=X.columns[0],
                                      var2=X.columns[0],
                                      binary_column=const.TARGET)

        X = DBSCANClustering(apply_grid_search=True).fit_transform(X)

        fig_pred = plot_two_variables(df=X,
                                      var1=X.columns[0],
                                      var2=X.columns[0],
                                      binary_column=const.PRED)
        ### PAREI AQUI ###
        # Pensar em como fazer o pré-processament com os label do DBSCAN
        # Fazer pré-processamento com o target também?
        # calcular as métricas e salvar no MLFlow -> Acho que é melhor salvar na chamada principal dessa função
        # pensar também em como salvar os logs antes de depois do pré-processamento
    
    def __get_norm_columns(self, X: pd.DataFrame):
        return list(X.columns).remove(const.TARGET)
