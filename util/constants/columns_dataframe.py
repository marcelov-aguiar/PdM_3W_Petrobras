# columns DataFrame df_instances
CLASS_CODE = "class_code"
INSTANCE_PATH = "instance_path"
OIL_WELL_NUMBER = "oil_well_number"

INDEX_NAME = "timestamp"

CLASSES_CODES = [1, 2, 5, 6, 7]

# Mapping of failure and non-failure.
MAPPING_TWO_CLASSES = {2: 1, 5: 1, 6: 1, 7: 1, 8: 1, 101: 1, 102: 1, 105: 1, 106: 1, 107: 1, 108: 1}

MAPPING_ALL_CLASSES = {101: 1, 102: 2, 105: 5, 106: 6, 107: 7, 108: 8}

TEST_SIZE = 0.3

VAL_SIZE = 0.2

RANDOM_STATE = 42

TARGET = "class"

BAD_COLUMNS = ["T-JUS-CKGL","P-PDG", "P-JUS-CKGL", "QGL"]

GOOD_COLUMNS = ["P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "class"]

N_COMPONENTS_PCA = 4

PRED="PRED"