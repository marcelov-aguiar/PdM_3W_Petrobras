import os
import sys

"""
Esse script é responsável por executar o mlflow para rastrear
os experimentos realizados.
"""

base_path = os.path.dirname(os.path.realpath(__file__))

path_utils = os.path.abspath(
    os.path.join(
        base_path,
        '..',
        '..',
        '..',
        '0_utilidades'))
sys.path.append(path_utils)

import util
from class_manipulates_path import ManipulatePath
from class_update_meta_mlflow import UpdateMetaMLFlow
util.init()
manipulate_tag = ManipulatePath()

path_mlflow = str(manipulate_tag.get_path_mlflow())

if os.path.isdir(path_mlflow):
    update_meta_mlflow = UpdateMetaMLFlow()
    update_meta_mlflow.process_update(path_mlflow)

os.system(f"mlflow ui --backend-store-uri file:///{path_mlflow} &")
