# 🏠 House Price Prediction - Local Training Guide

This directory contains standalone Python scripts extracted from the Jupyter notebook, allowing you to train the house price prediction model on your local laptop.

## 📁 Project Structure

```
training/
├── modules/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration & constants
│   ├── cache_manager.py         # Training result caching
│   ├── transformers.py          # Custom sklearn transformers
│   ├── feature_engineering.py   # Feature engineering pipeline
│   └── models.py                # Model definitions & training
├── outputs/                      # Generated outputs (created automatically)
│   ├── training_cache/          # Cached training results
│   ├── logs/                    # Training logs & results
│   └── mlruns/                  # MLflow tracking (optional)
├── train.py                     # Main training script
├── inference.py                 # Prediction script
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Optional: For MLflow tracking
pip install mlflow
```

### 2. Prepare Data

Place your training data at:
```
../data/train-house-prices-advanced-regression-techniques.csv
```

Or update the `data_path` in `train.py`.

### 3. Run Training

```bash
# Navigate to training directory
cd training

# Run baseline model training
python train.py
```

### 4. Make Predictions

```bash
# After training completes, run inference on new data
python inference.py ../data/test.csv

# Or specify custom model path
python inference.py ../data/test.csv ./outputs/logs/final_model.pkl
```

## 🔧 Configuration

Edit `modules/config.py` to customize:

- **RANDOM_STATE**: Random seed for reproducibility (default: 42)
- **MODEL_PARAMS**: Hyperparameters for each model
- **TARGET_ENCODER_FEATURES**: Features to use with target encoding
- **RARE_POOLER_MIN_COUNT**: Minimum frequency to pool rare categories (default: 15)
- **QUANTILE_TRANSFORMER_N_QUANTILES**: Number of quantiles for scaling (default: 200)

## 📊 Features

### Model Selection
The training pipeline evaluates 5+ baseline models:
- **RF**: Random Forest Regressor
- **Ridge**: Ridge Regression
- **Lasso**: Lasso Regression
- **ENet**: Elastic Net
- **SVR**: Support Vector Regressor
- **XGB**: XGBoost (if installed)
- **Cat**: CatBoost (if installed)
- **LGBM**: LightGBM (if installed)

### Feature Engineering
Custom features created:
- **TotalSF**: Total square footage
- **TotalBath**: Total bathrooms
- **HouseAge**: Years since construction
- **RemodAge**: Years since remodeling
- **Quality_Area_Interaction**: Quality × Living Area
- **Neighborhood_BldgType**: Location × Building type interaction
- Cyclical encoding for month of sale

### Data Preprocessing
- **Ordinal Mapping**: Maintains feature hierarchy
- **Missingness Indicators**: Flags for missing values
- **Rare Category Pooling**: Groups rare categories
- **Target Encoding**: Categorical feature encoding
- **Quantile Transformation**: Robust scaling

## 📈 Output Files

After training, check `outputs/` folder:

```
outputs/
├── logs/
│   ├── baseline_results.csv                    # Model comparison
│   ├── final_prediction_scatter.png            # Predictions visualization
│   ├── baseline_model_comparison.png           # Model performance chart
│   ├── final_house_price_model.pkl             # Trained model
│   ├── best_model_config.yaml                  # Model configuration
│   ├── best_model_config.json                  # Model config (JSON)
│   └── final_ranking.csv                       # Final model ranking
├── training_cache/
│   ├── baseline_results_local.pkl              # Cached baseline results
│   └── baseline_results_local_meta.json        # Metadata
└── mlruns/ (if MLflow enabled)
    └── House_Price_Prediction_Local/           # Experiment tracking
```

## 💾 Using Cached Results

The training system caches results to speed up re-runs:

```python
from modules.cache_manager import MLflowTrainingCacheManager

cache = MLflowTrainingCacheManager('./outputs/training_cache')

# Check if cached results exist
if cache.results_exist('baseline_results_local'):
    results, metadata = cache.load_results('baseline_results_local')
    print("Loaded from cache!")

# Clear cache if needed
cache.clear_cache('baseline_results_local')
```

## 🎯 Customization Examples

### 1. Train Specific Models Only

Edit `train.py`:
```python
# In base_models_dict return statement, keep only desired models
models = {
    "RF": RandomForestRegressor(...),
    "Ridge": Ridge(...),
    # Remove others
}
```

### 2. Adjust Train-Test Split

Edit `train.py`, in `prepare_data()`:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE  # Change 0.2 to 0.15
)
```

### 3. Enable MLflow Tracking

MLflow will automatically track experiments if installed. Access UI:
```bash
mlflow ui --backend-store-uri file:./outputs/mlruns
# Opens at http://localhost:5000
```

## ❌ Troubleshooting

### "Module not found" Error
```bash
# Ensure you're in the training directory
cd training

# Or add parent directory to Python path
export PYTHONPATH=$PYTHONPATH:..
python train.py
```

### Data File Not Found
```bash
# Check data path - update in train.py or adjust path
# Default expected location:
../data/train-house-prices-advanced-regression-techniques.csv
```

### Out of Memory
- Reduce `n_jobs` in `train.py` from `-1` to `1` or `2`
- Reduce `QUANTILE_TRANSFORMER_SUBSAMPLE` in `config.py`

### Missing Optional Dependencies
```bash
# Install boosting libraries (optional)
pip install xgboost catboost lightgbm optuna
```

## 📝 Model Output

After training, the best model's configuration is saved:

**best_model_config.yaml**:
```yaml
model_info:
  name: XGB              # Model name
  type: Single           # Single or Ensemble
  timestamp: 2024-...
  model_version: 1.0

performance:
  cv_rmse: 20500.45      # Cross-validation RMSE
  test_rmse: 21000.12    # Test set RMSE
  test_r2: 0.8934        # R² score

hyperparameters:
  n_estimators: 1500
  learning_rate: 0.05
  ...
```

## 🔄 Training Pipeline Flow

```
1. Load Data
    ↓
2. Prepare Data (Train-Test Split)
    ↓
3. Feature Engineering
    - Add domain features
    - Ordinal encoding
    - Target encoding
    - One-hot encoding
    ↓
4. Model Training
    - Train each model
    - Cross-validation
    - Test set evaluation
    ↓
5. Results & Caching
    - Save best model
    - Cache results
    - Log metrics
    ↓
6. Done! ✅
```

## 📚 Module Functions

### `modules.config`
- Configuration and constants
- Paths for data, logs, cache

### `modules.transformers`
- `OrdinalMapper`: Maps ordinal categories to numbers
- `MissingnessIndicator`: Creates binary flags for missing values
- `RarePooler`: Groups rare categories
- `TargetEncoder`: Target-based categorical encoding
- `DataSanitizer`: Handles infinite/NaN values

### `modules.feature_engineering`
- `add_domain_features()`: Creates domain-specific features
- `make_feature_space()`: Full preprocessing pipeline
- `build_feature_lists()`: Identifies feature types

### `modules.models`
- `base_models_dict()`: Initialize baseline models
- `evaluate_all_models()`: Parallel model evaluation
- `get_metrics()`: Calculate RMSE and R²

## 📞 Notes for Developers

- All code is modular and reusable
- Custom transformers follow sklearn API
- Pipeline compatible with scikit-learn
- Easily extendable for additional models
- Caching system for efficient development

---

**Happy Training! 🚀**

For questions or issues, check the logs in `outputs/logs/` or review the original notebook.
