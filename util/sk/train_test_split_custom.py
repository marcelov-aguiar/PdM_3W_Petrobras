from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sk.base_transform import BaseTransform


class TrainTestSplitCustom(BaseTransform):
    def __init__(self,
                 input_columns: List[str],
                 target: str,
                 test_size: float = 0.3,
                 stratify: bool = False,
                 random_state: float = 42):
        self.input_columns = input_columns
        self.target = target
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state

    def fit(self, X:pd.DataFrame=None, y=None):
        return self

    def transform(self, df_data: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.target in self.input_columns:
            self.input_columns.remove(self.target)

        X = df_data[self.input_columns].copy()
        y = df_data[self.target].copy()

        if self.stratify:
            y_ = y
        else:
            y_ = None

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self.test_size,
                                                            stratify=y_,
                                                            random_state=self.random_state)

        return X_train, X_test, y_train, y_test
