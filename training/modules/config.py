# ============================================================================
# Configuration and Constants
# ============================================================================
import numpy as np
from pathlib import Path

# Random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Base paths (Adjusted for local machine - not Colab)
BASE_PATH = Path("./outputs")
CACHE_DIR = BASE_PATH / "training_cache"
LOGS_DIR = BASE_PATH / "logs"
MLFLOW_DIR = BASE_PATH / "mlruns"

# Create directories
for directory in [CACHE_DIR, LOGS_DIR, MLFLOW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Ordinal mapping for categorical features
ORDINAL_MAP_CANONICAL = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
    "GarageQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["NA", "Fa", "TA", "Gd", "Ex"],
    "Fence": ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "PavedDrive": ["N", "P", "Y"],
    "Street": ["Grvl", "Pave"],
    "Alley": ["NA", "Grvl", "Pave"],
    "CentralAir": ["N", "Y"]
}

# Target Encoder features
TARGET_ENCODER_FEATURES = ["Neighborhood", "MSZoning", "Exterior1st", "Exterior2nd", "SaleCondition", "BldgType"]
TARGET_ENCODER_ALPHA = 30.0

# Rare Pooler settings
RARE_POOLER_MIN_COUNT = 15

# Quantile Transformer settings
QUANTILE_TRANSFORMER_N_QUANTILES = 200
QUANTILE_TRANSFORMER_SUBSAMPLE = 200000

# Model hyperparameters (baseline)
MODEL_PARAMS = {
    "RandomForest": {
        "n_estimators": 500,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "ElasticNet": {
        "alpha": 0.01,
        "l1_ratio": 0.5,
        "max_iter": 20000,
        "random_state": RANDOM_STATE
    },
    "Ridge": {
        "alpha": 10.0,
        "random_state": RANDOM_STATE
    },
    "Lasso": {
        "alpha": 0.0005,
        "max_iter": 20000,
        "random_state": RANDOM_STATE
    },
    "SVR": {
        "C": 10.0,
        "epsilon": 0.1
    },
    "XGBoost": {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": 1
    },
    "CatBoost": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_state": RANDOM_STATE,
        "verbose": False
    },
    "LightGBM": {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "verbosity": -1
    }
}

# MLflow configuration
USE_MLFLOW = True
MLFLOW_EXPERIMENT_NAME = "House_Price_Prediction"
