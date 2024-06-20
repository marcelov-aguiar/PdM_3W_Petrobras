from class_manipulates_path import ManipulatePath
from class_report_multi_dataset import ReportMultiDataset


manipulate_path = ManipulatePath()
path_raw_data = manipulate_path.get_path_raw_data()

list_file_path = list(path_raw_data.rglob("*.csv"))
 
report = ReportMultiDataset(str(manipulate_path.get_path_report_raw_data()))
report.process_all_csv(list_file_path)

print("Final teste")