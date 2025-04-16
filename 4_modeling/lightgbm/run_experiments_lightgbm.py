import util
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
import constants.columns_dataframe as const
from sk.train_test_split_custom import TrainTestSplitCustom
from sk.norm_standard_scaler import NormStandardScaler
from class_manipulates_path import ManipulatePath

manipulate_path = ManipulatePath()

def evaluate_lgbm(X: pd.DataFrame, y: pd.Series, kf: KFold):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        lgbm = LGBMClassifier()
        lgbm.fit(X_train, y_train)
        y_pred = lgbm.predict(X_test)
        
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

kf = KFold(n_splits=5, shuffle=False)

accuracy, precision, recall, f1, accuracy_std = evaluate_lgbm(X_train_all_norm, y_train_all, kf)
results = [{'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, "Accuracy Std": accuracy_std}]
results_df = pd.DataFrame(results)
print(results_df)

scaler = NormStandardScaler(X_train.columns)
X_train_all_norm = scaler.fit_transform(X_train_all)
X_test_norm = scaler.transform(X_test)

lgbm = LGBMClassifier()
lgbm.fit(X_train_all_norm, y_train_all)

y_pred = lgbm.predict(X_test_norm)

print("Acurácia do modelo", accuracy_score(y_test, y_pred))
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
        
for i in range(len(precision)):
    print(f'Classe {y_test.unique()[i]}:')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'F1-score: {f1[i]}\n')


print("Sucesso")
