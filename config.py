# config.py

from pathlib import Path
import os

# Automatically detect if running in Colab
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
# BASE_PATH always resolves to the project root (where this config.py lives)
BASE_PATH = Path(__file__).resolve().parent

# Directories — inputs and outputs
# On Colab, point to Google Drive so data persists across sessions.
# Locally, everything lives under the repo root.
if IN_COLAB:
    DRIVE_INPUT   = Path("/content/drive/MyDrive/Environmental_DL_Project/GroundwaterFlowGNN-main")
    DRIVE_OUTPUT  = Path("/content/drive/MyDrive/Environmental_DL_Project/GroundwaterFlowGNN")
    INPUT_DIR = DRIVE_INPUT / "data" / "input"
    PREPROCESSED_DIR = DRIVE_OUTPUT / "data" / "preprocessed"
    GENERATED_GRAPHS = DRIVE_OUTPUT / "generated_graphs"
    SCATTER_PLOTS = DRIVE_OUTPUT / "scatterplots"
    SAVED_MODELS_DIR = DRIVE_OUTPUT / "saved_models"
    TRAINING_RESULTS_DIR = DRIVE_OUTPUT / "training_results"
    OUTPUTS_DIR = DRIVE_OUTPUT / "outputs"
else:
    INPUT_DIR = BASE_PATH / "data" / "input"
    PREPROCESSED_DIR = BASE_PATH / "data" / "preprocessed"
    GENERATED_GRAPHS = BASE_PATH / "generated_graphs"
    SCATTER_PLOTS = BASE_PATH / "scatterplots"
    SAVED_MODELS_DIR = BASE_PATH / "saved_models"
    TRAINING_RESULTS_DIR = BASE_PATH / "training_results"
    OUTPUTS_DIR = BASE_PATH / "outputs"
PIEZO_CSV_DIR = INPUT_DIR / "piezometers/csv/csv"
TRAINING_SUMMARIES = TRAINING_RESULTS_DIR / "Summaries"
RUN_PLOTS_AND_RESULTS = TRAINING_RESULTS_DIR / "Individual_Run_Results"

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
PUMP_COHERENCE_WEIGHTS = INPUT_DIR / "wells/pump_weights_coherence.csv"
if IN_COLAB:
    HYDRAULIC_RESISTANCE_DIR = DRIVE_OUTPUT / "data" / "Shortest_Path_Graphs"
    REGIS_DIR = DRIVE_OUTPUT / "data" / "REGIS II Study Area Raster Data"
else:
    HYDRAULIC_RESISTANCE_DIR = BASE_PATH / "data" / "Shortest_Path_Graphs"
    REGIS_DIR = BASE_PATH / "data" / "REGIS II Study Area Raster Data"
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
NODE_K_VALUES = PREPROCESSED_DIR / "node_k_values.csv"
FEATURE_DISTANCE_7D = PREPROCESSED_DIR / "feature_distance_7d.npy"

