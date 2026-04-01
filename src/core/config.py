"""
Configuration module for House Price Prediction project.
Defines all constants, paths, and configuration parameters.
"""

import os
from pathlib import Path
import logging

# ============================================================================
# BASE PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
EXPLAINER_DIR = ARTIFACTS_DIR / "explainers"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Create directories if they don't exist
for directory in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, SCALERS_DIR, EXPLAINER_DIR, LOGS_DIR, CONFIGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_URL = "https://drive.google.com/uc?id=1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd"
TRAIN_DATA_FILE = DATA_DIR / "train-house-prices-advanced-regression-techniques.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# MLFLOW CONFIGURATION
# ============================================================================
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = f"file:{MLFLOW_DIR}"
MLFLOW_EXPERIMENT_NAME = "House_Price_Prediction"

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================
CACHE_DIR = PROJECT_ROOT / "training_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================
# Cross-validation splits
CV_SPLITS = 5

# Model hyperparameters defaults
RF_N_ESTIMATORS = 500
RF_MAX_FEATURES = "sqrt"

XGB_N_ESTIMATORS = 2000
XGB_LEARNING_RATE = 0.03
XGB_MAX_DEPTH = 4

CAT_N_ESTIMATORS = 1000
CAT_LEARNING_RATE = 0.05
CAT_DEPTH = 6

LGBM_N_ESTIMATORS = 2000
LGBM_LEARNING_RATE = 0.03

# Linear models
ELASTICNET_ALPHA = 0.01
ELASTICNET_L1_RATIO = 0.5
RIDGE_ALPHA = 10.0
LASSO_ALPHA = 0.0005

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================


SCHEMA_TO_DATA_MAPPING = {
    "FirstFlrSF": "1stFlrSF",
    "SecondFlrSF": "2ndFlrSF", 
    "ThreeSsnPorch": "3SsnPorch",
}

# Ordinal feature mapping
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

# Target encoding
TARGET_ENCODER_COLS = ["Neighborhood", "MSZoning", "Exterior1st", "Exterior2nd", 
                        "SaleCondition", "BldgType", "Neighborhood_BldgType"]
TARGET_ENCODER_ALPHA = 30.0

# Rare pooler
RARE_POOLER_MIN_COUNT = 15

# Quantile transformer
QUANTILE_N_QUANTILES = 200
QUANTILE_SUBSAMPLE = 200000

# Domain features to generate
DOMAIN_FEATURES = [
    "TotalSF", "TotalBath", "HouseAge", "RemodAge", "IsRemodeled",
    "Has2ndFlr", "TotalPorchSF", "Quality_Area_Interaction", 
    "MoSold_sin", "MoSold_cos", "Loc_Type", "LotArea_clip"
]

# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ============================================================================
OPTUNA_N_TRIALS = 15
OPTUNA_N_JOBS = 1

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "app.log"
