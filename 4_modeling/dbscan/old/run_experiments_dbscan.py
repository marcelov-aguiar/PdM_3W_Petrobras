import util
import constants.columns_dataframe as const
from sk.search_file_csv import SearchFileCSV
from class_manipulates_path import ManipulatePath
from class_format_data import FormatData
from modeling_dbscan import ModelingDBSCAN
manipulate_path = ManipulatePath()


class ClassNull(Exception):
    pass

df_instances = SearchFileCSV(manipulate_path.get_path_raw_data(),
                             const.CLASSES_CODES
                            ).get_csv_path()


for csv_path in df_instances[const.INSTANCE_PATH]:
    df_raw_data = FormatData.read_data(csv_path,
                                       const.INDEX_NAME)
    #TODO: Fazer report
    try:
        ModelingDBSCAN().run_modeling(df_raw_data, csv_path, const.TARGET)
    except ClassNull:
        #TODO: Implementar log
        continue
    except:
        raise

    print("Sucess")