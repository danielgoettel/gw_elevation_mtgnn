# config.py

from pathlib import Path
import os

# Automatically detect if running in Colab
IN_COLAB = "COLAB_GPU" in os.environ
if IN_COLAB:
    BASE_PATH = Path("/content/drive/MyDrive/Environmental_DL_Project/GroundwaterFlowGNN-main")
else:
    BASE_PATH = Path(__file__).resolve().parent

# Directories
DATA_DIR = BASE_PATH / "data"
INPUT_DIR = DATA_DIR / "input"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
PIEZO_CSV_DIR = INPUT_DIR / "piezometers/csv/csv"
GENERATED_GRAPHS = BASE_PATH / "generated_graphs"
SCATTER_PLOTS = BASE_PATH / "scatterplots"
TRAINING_SUMMARIES = BASE_PATH / "training_results/Summaries"
RUN_PLOTS_AND_RESULTS = BASE_PATH / "training_results/Individual_Run_Results"

#JSON FILE PATH

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
PUMP_DISTANCES = INPUT_DIR / "wells/wellfield_to_obswell_distances.csv"
RANDOM_FOREST_TRAINING_DATA = INPUT_DIR / "piezo_only_rf_training_data.csv"

#Pickl files for RF Training Importances
RF_TRAINED_ALL = PREPROCESSED_DIR / "raw_importances.pkl"
RF_TRAINED_PIEZOS_ONLY = PREPROCESSED_DIR / "raw_piezo_only_importances.pkl"

# Output files
ADJ_MATRIX_PATH = PREPROCESSED_DIR / "adj_matrix.pt"
STATIC_FEATURES_PATH = PREPROCESSED_DIR / "static_features.pt"
PROCESSED_DATA_FILE = PREPROCESSED_DIR / "processed_data.pkl"
COLUMN_NAMES_REAL = PREPROCESSED_DIR / "column_names_real.txt"
COLUMN_NAMES_SYN = PREPROCESSED_DIR / "column_names_synthetic.txt"

