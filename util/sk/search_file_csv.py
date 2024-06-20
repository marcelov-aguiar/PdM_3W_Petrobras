from pathlib import Path
from typing import List
import pandas as pd
import re
import util
import constants.columns_dataframe as const

class SearchFileCSV():
    # columns DataFrame df_instances
    class_code = const.CLASS_CODE
    instance_path = const.INSTANCE_PATH
    oil_well_number = const.OIL_WELL_NUMBER

    def __init__(self,
                 raw_path: Path,
                 classes_codes: List[int],
                 instance_real: bool = True,
                 instance_simulated: bool = False,
                 instance_drawn: bool = False) -> None:
        self.raw_path = raw_path
        self.instance_real = instance_real
        self.instance_simulated = instance_simulated
        self.instance_drawn = instance_drawn
        self.classes_codes = classes_codes

    def get_csv_path(self):
        df_instances = self._get_instances()
        
        df_instances = self._set_oil_well(df_instances)

        return df_instances
    
    def _get_instances(self) -> pd.DataFrame:
        # Gets all real instances but maintains only those with any type of undesirable event
        df_instances = pd.DataFrame(self._class_and_file_generator(self.raw_path, 
                                                               real=self.instance_real,
                                                               simulated=self.instance_simulated, 
                                                               drawn=self.instance_drawn),
                                      columns=[self.class_code, self.instance_path])
        df_instances = df_instances.loc[df_instances.iloc[:,0].isin(self.classes_codes)].reset_index(drop=True)

        return df_instances

    def _class_and_file_generator(self,
                                  data_path: Path,
                                  real=False,
                                  simulated=False,
                                  drawn=False):
        for class_path in data_path.iterdir():
            if class_path.is_dir():
                class_code = int(class_path.stem)
                for instance_path in class_path.iterdir():
                    if (instance_path.suffix == '.csv'):
                        if (simulated and instance_path.stem.startswith('SIMULATED')) or \
                           (drawn and instance_path.stem.startswith('DRAWN')) or \
                           (real and (not instance_path.stem.startswith('SIMULATED')) and \
                           (not instance_path.stem.startswith('DRAWN'))):
                            yield class_code, instance_path
    
            
    def _set_oil_well(self, df_instances) -> pd.DataFrame:
        list_group = []
        for path in df_instances[self.instance_path].values:
            path = str(path)
            list_group.append(int(self._extract_well_number(path)))

            df_instances[self.oil_well_number] = pd.Series(list_group)
        return df_instances

    def _extract_well_number(self,
                             path: str):
        match = re.search(r'WELL-(\d+)_', path)
        if match:
            return match.group(1)
        else:
            return None

if __name__ == "__main__":
    from class_manipulates_path import ManipulatePath
    manipulate_path = ManipulatePath()
    
    search_csv = SearchFileCSV(
        manipulate_path.get_path_raw_data(),
        const.CLASSES_CODES
    )
    df_instances = search_csv.get_csv_path()
    print("Sucess")