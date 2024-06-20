import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
import util
from class_manipulates_path import ManipulatePath
util.init()
manipulate_tag = ManipulatePath()

path_mlflow = manipulate_tag.get_path_mlflow()
mlflow.set_tracking_uri(f"file:///{str(path_mlflow)}") ###
mlflow.set_experiment('classificacao de petalas')
dataset = load_iris(as_frame=True)
features, target = dataset['data'].values, dataset['target'].values

with mlflow.start_run(run_name='iris logistic kfold') as parent_run:

    metrics = defaultdict(list)
    folder = StratifiedKFold(n_splits=3, shuffle=True)
    fold = 1
    for train_idx, test_idx in folder.split(features, target):
        with mlflow.start_run(nested=True, run_name = f'fold {fold}') as child_run:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('logistic', LogisticRegression(solver='saga'))
            ])
            train_features = features[train_idx]
            test_features = features [test_idx]
            train_target = target[train_idx]
            test_target = target[test_idx]

            pipe.fit(train_features, train_target)

            fold_acc = pipe.score(test_features, test_target) #returns mean acc
            pred_proba = pipe.predict_proba(test_features)
            one_hot = OneHotEncoder(sparse=False).fit_transform(test_target.reshape(-1, 1))
            fold_auc = roc_auc_score(one_hot, pred_proba, multi_class='ovr')

            metrics['acc'].append(fold_acc)
            metrics['auc'].append(fold_auc)

            cm = ConfusionMatrixDisplay.from_estimator(pipe, test_features, test_target, cmap='Blues', values_format='.3g')

            mlflow.log_metric('acc', fold_acc)
            mlflow.log_metric('auc', fold_auc)
            mlflow.log_figure(cm.figure_, artifact_file='confusion_matrix.png')
            input_example = dataset['data'].iloc[:10]
            #signature = infer_signature(input_example, pipe.predict(infer_signature))
            input_schema = Schema([
                ColSpec("double", "sepal length (cm)"),
                ColSpec("double", "sepal width (cm)"),
                ColSpec("double", "petal length (cm)"),
                ColSpec("double", "petal width (cm)"),
                ])
            output_schema = Schema([ColSpec("double")])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            mlflow.sklearn.log_model(pipe, 'model', input_example=input_example, signature=signature)
            fold+=1
        
        
        mlflow.log_metric('acc', fold_acc, step=fold)
        mlflow.log_metric('auc', fold_auc, step=fold)
    
    mlflow.sklearn.log_model(pipe, 'model', input_example=input_example, signature=signature)  
    acc = np.array(metrics['acc'])
    mlflow.log_metric('mean_acc', acc.mean())
    mlflow.log_metric('std_acc', acc.std())

    auc = np.array(metrics['auc'])
    mlflow.log_metric('mean_auc', auc.mean())
    mlflow.log_metric('std_auc', auc.std())
