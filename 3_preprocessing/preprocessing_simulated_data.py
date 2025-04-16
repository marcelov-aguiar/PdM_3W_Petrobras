import util
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import constants.columns_dataframe as const
from sk.search_file_csv import SearchFileCSV
from class_manipulates_path import ManipulatePath
from class_format_data import FormatData
from sk.drop_columns import DropColumns
from sk.drop_columns_low_variance import DropLowVariance
from sk.drop_index_duplicate import DropIndexDuplicate
from sk.drop_nan import DropNaN
from sk.replace_column_dataframe import ReplaceColumnDataFrame
from sk.drop_index_duplicate_by_column import DropIndexDuplicateByColumn

manipulate_path = ManipulatePath()

class ClassNull(Exception):
    pass

df_instances = SearchFileCSV(raw_path=manipulate_path.get_path_raw_data(),
                             classes_codes=const.CLASSES_CODES,
                             instance_real=False,
                             instance_simulated=True,
                             instance_drawn=False
                            ).get_csv_path()

pipe_preprocessing = Pipeline([
    ("DropColumns", DropColumns(const.BAD_COLUMNS)),
    ("DropIndexDuplicateByColumn", DropIndexDuplicateByColumn(const.TARGET)),
    ("DropLowVariance",  DropLowVariance(exception_columns=[const.TARGET],
                                         min_variance=0.001)),
    ("DropIndexDuplicate", DropIndexDuplicate()),
    ("DropNaN", DropNaN())
])

df_preprocessing = pd.DataFrame()
i = 0
tam = df_instances.shape[0]
for csv_path in df_instances[const.INSTANCE_PATH]:
    i = i + 1
    print(f"{i} de {tam}")
    print(f"{csv_path}")
    df_raw_data = FormatData.read_data(csv_path, const.INDEX_NAME)

    # #TODO: Fazer report
    df_raw_data = pipe_preprocessing.transform(df_raw_data)
    #TODO: Fazer report

    try:
        df_raw_data[const.GOOD_COLUMNS] = df_raw_data[const.GOOD_COLUMNS].copy()
        if df_raw_data.empty:
            continue
    except KeyError:
        continue
    df_preprocessing = pd.concat([df_preprocessing, df_raw_data])
    

print("Removendo indices duplicados")
# remove indices duplicados com classes diferentes!
df_preprocessing = df_preprocessing.drop_duplicates()

# transforma classe em falha e não falha
df_preprocessing = ReplaceColumnDataFrame(const.TARGET, const.MAPPING_ALL_CLASSES).transform(df_preprocessing)

df_preprocessing.to_parquet(manipulate_path.get_path_preprocessing_simulated_data())