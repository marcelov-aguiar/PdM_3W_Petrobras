import util
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import constants.columns_dataframe as const
from sk.search_file_csv import SearchFileCSV
from class_manipulates_path import ManipulatePath
from class_format_data import FormatData
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from modeling_knn import ModelingKNN
from sk.drop_columns import DropColumns
from sk.drop_columns_null_data import DropNullData
from sk.drop_columns_low_variance import DropLowVariance
from sk.drop_index_duplicate import DropIndexDuplicate
from sk.drop_nan import DropNaN
from sk.preprocessor import Preprocessor
from sk.replace_column_dataframe import ReplaceColumnDataFrame
from sk.train_test_split_custom import TrainTestSplitCustom
from sk.drop_index_duplicate_by_column import DropIndexDuplicateByColumn
from sk.norm_standard_scaler import NormStandardScaler

manipulate_path = ManipulatePath()

def evaluate_knn_for_k(X: pd.DataFrame, y: pd.Series, k: int, kf: KFold):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
    
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s), np.std(accuracies)


df_preprocessing = pd.read_parquet(manipulate_path.get_path_preprocessing_real_data_all_classes())

X_train_all, X_test, y_train_all, y_test = TrainTestSplitCustom(
    input_columns=list(df_preprocessing.columns),
    target=const.TARGET,
    test_size=const.TEST_SIZE,
    stratify=True).transform(df_preprocessing)
 
X_train, X_val, y_train, y_val = train_test_split(X_train_all,
                                                  y_train_all,
                                                  test_size=const.VAL_SIZE,
                                                  stratify=y_train_all,
                                                  random_state=const.RANDOM_STATE)

scaler = NormStandardScaler(X_train.columns)
X_train_norm = scaler.fit_transform(X_train)
X_train_all_norm = scaler.transform(X_train_all)

k_values = [3, 5, 7, 9]
results = []

kf = KFold(n_splits=5, shuffle=False)

for k in k_values:
    accuracy, precision, recall, f1, accuracy_std = evaluate_knn_for_k(X_train_all_norm, y_train_all, k, kf)
    results.append({'K': k, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, "Accuracy Std": accuracy_std})

results_df = pd.DataFrame(results)
print(results_df)
max_f1_index = results_df['F1-Score'].idxmax()

# Obter o valor de K correspondente ao maior F1-Score
k_with_max_f1 = results_df.loc[max_f1_index, 'K']

scaler = NormStandardScaler(X_train.columns)
X_train_all_norm = scaler.fit_transform(X_train_all)
X_test_norm = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=k_with_max_f1)
knn.fit(X_train_all_norm, y_train_all)

y_pred = knn.predict(X_test_norm)

print("Acurácia do modelo", accuracy_score(y_test, y_pred))
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
        
for i in range(len(precision)):
    print(f'Classe {y_test.unique()[i]}:')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'F1-score: {f1[i]}\n')


"""
print("Avaliação com os dados simulados")
df_instances = SearchFileCSV(raw_path=manipulate_path.get_path_raw_data(),
                             classes_codes=const.CLASSES_CODES,
                             instance_real=False,
                             instance_simulated=True
                            ).get_csv_path()

df_preprocessing = pd.DataFrame()
for csv_path in df_instances[const.INSTANCE_PATH]:
    df_raw_data = FormatData.read_data(csv_path, const.INDEX_NAME)

    # #TODO: Fazer report
    df_raw_data = pipe_preprocessing.transform(df_raw_data)
    #TODO: Fazer report

    try:
        df_raw_data[const.GOOD_COLUMNS] = df_raw_data[const.GOOD_COLUMNS].copy()
    except KeyError:
        continue
    df_preprocessing = pd.concat([df_preprocessing, df_raw_data])


# remove indices duplicados com classes diferentes!
df_preprocessing = pipe_preprocessing.transform(df_preprocessing)

# transforma classe em falha e não falha
df_preprocessing = ReplaceColumnDataFrame(const.TARGET, const.MAPPING_ALL_FAILURE).transform(df_preprocessing)

X = df_preprocessing.drop(columns=[const.TARGET])

y = df_preprocessing[const.TARGET]

X = scaler.transform(X)

knn = KNeighborsClassifier(n_neighbors=3) # retirar
y_pred = knn.predict(X)

precision = precision_score(y, y_pred, average=None)
recall = recall_score(y, y_pred, average=None)
f1 = f1_score(y, y_pred, average=None)
        
for i in range(len(precision)):
    print(f'Classe {y.unique()[i]}:')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'F1-score: {f1[i]}\n')
"""
print("Sucess")