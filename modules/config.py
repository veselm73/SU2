import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random Seed
SEED = 42

# Synthetic Data Config
PATCH_SIZE = 128
SIM_CONFIG = {
    "na": 1.49,
    "wavelength": 512,
    "px_size": 0.07,
    "wiener_parameter": 0.1,
    "apo_cutoff": 2.0,
    "apo_bend": 0.9
}
MIN_CELLS = 5
MAX_CELLS = 15

# Training Config
TRAIN_SAMPLES = 500
VAL_SAMPLES = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.1
EPOCHS = 200
PATIENCE = 5

# Paths (Colab specific defaults)
VAL_DATA_PATH = "/content/val_data"
VAL_TIF_PATH = f"{VAL_DATA_PATH}/val.tif"
VAL_CSV_PATH = f"{VAL_DATA_PATH}/val.csv"
