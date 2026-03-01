# config.py

from pathlib import Path
import os

# Automatically detect if running in Colab
IN_COLAB = "COLAB_GPU" in os.environ
# BASE_PATH always resolves to the project root (utils/ -> parent)
BASE_PATH = Path(__file__).resolve().parent.parent

# Experiment tag — change this to write outputs to new folders
EXPERIMENT = "Independent_Study"

# Directories — inputs (shared across experiments)
DATA_DIR = BASE_PATH / "data"
INPUT_DIR = DATA_DIR / "input"
PIEZO_CSV_DIR = INPUT_DIR / "piezometers/csv/csv"

# Directories — outputs (experiment-specific)
PREPROCESSED_DIR = DATA_DIR / f"preprocessed_{EXPERIMENT}"
SAVED_MODELS_DIR = BASE_PATH / f"saved_models_{EXPERIMENT}"
TRAINING_RESULTS_DIR = BASE_PATH / f"training_results_{EXPERIMENT}"

# Metadata files
PIEZO_METADATA = INPUT_DIR / "piezometers/piezometer_metadata.csv" #validated
PUMP_METADATA = INPUT_DIR / "wells/pump_metadata.csv"
EVAP_METADATA = INPUT_DIR / "meteo_metadata_and_timeseries/evap_metadata.csv" #validated
PREC_METADATA = INPUT_DIR / "meteo_metadata_and_timeseries/prec_metadata.csv" #validated
RIVER_METADATA = INPUT_DIR / "river/rivers_metadata.csv" #validated
PUMPING_WELLS_PATH = INPUT_DIR / "wells/pump_daily.csv"
PRECIP_PATH = INPUT_DIR / "meteo_metadata_and_timeseries/precipitation.csv" #validated
EVAP_PATH = INPUT_DIR / "meteo_metadata_and_timeseries/evaporation.csv" #validated
RIVER_PATH = INPUT_DIR / "river/river_daily.csv" #validated

PIEZO_LAYER_INFORMATION = INPUT_DIR / "piezometers/piezometer_layer_information.csv"

RANDOM_FOREST_TRAINING_DATA = INPUT_DIR / "rf_training_data.csv"

# Output files
ADJ_MATRIX_PATH = PREPROCESSED_DIR / "adj_matrix.pt"
STATIC_FEATURES_PATH = PREPROCESSED_DIR / "static_features.pt"
PROCESSED_DATA_FILE = PREPROCESSED_DIR / "processed_data.pkl"
COLUMN_NAMES_REAL = PREPROCESSED_DIR / "column_names_real.txt"
COLUMN_NAMES_SYN = PREPROCESSED_DIR / "column_names_synthetic.txt"

