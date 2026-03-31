# 💡 Tips & Tricks for Local Development

## ⚡ Performance Optimization

### 1. Reduce Training Time

#### Use fewer models
```python
# In train.py, modify base_models_dict
models = {
    "RF": RandomForestRegressor(...),
    "Ridge": Ridge(...),
}
# Remove other models
```

#### Reduce cross-validation folds
```python
# In models.py, change cv splits
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # Instead of 5
```

#### Reduce parallel jobs (if memory constrained)
```python
# In train.py
evaluate_all_models(..., n_jobs=2)  # Instead of -1
```

#### Skip quantile transformation
```python
# In feature_engineering.py, remove quantile transformer
num_cont_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    # Remove: ("qntl", QuantileTransformer(...)),
])
```

### 2. Enable GPU Acceleration

#### XGBoost with GPU
```python
# In modules/config.py
"XGBoost": {
    "gpu_id": 0,
    "tree_method": "gpu_hist",
    ...
}
```

#### LightGBM with GPU
```python
"LightGBM": {
    "device_type": "gpu",
    "gpu_platform_id": 0,
    ...
}
```

#### CatBoost with GPU
```python
"CatBoost": {
    "task_type": "GPU",
    "gpu_devices": "0:1",  # Multiple GPUs
    ...
}
```

### 3. Monitor Memory Usage

```python
# Add to train.py
import psutil

def check_memory():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {mem:.0f} MB")

# Call periodically
check_memory()
```

---

## 🎯 Development Tricks

### 1. Test on Small Data First

```python
# In train.py main()
# Reduce data size for testing
if TESTING_MODE:
    X_train = X_train.iloc[:1000]  # 1000 samples
    y_train = y_train.iloc[:1000]
```

### 2. Save Intermediate Results

```python
# Cache data at different stages
cache.save_results(X_train, 'X_train_processed')
cache.save_results(feature_pipe, 'feature_pipe_fitted')

# Load when needed
X_train, _ = cache.load_results('X_train_processed')
feature_pipe, _ = cache.load_results('feature_pipe_fitted')
```

### 3. Debug Feature Engineering

```python
# Add to feature_engineering.py
def debug_features(df_original, df_processed):
    print("Feature summary:")
    print(f"  Original shape: {df_original.shape}")
    print(f"  Processed shape: {df_processed.shape}")
    print(f"  New features: {df_processed.shape[1] - df_original.shape[1]}")
    print(f"  Missing values: {df_processed.isnull().sum().sum()}")
    return df_processed

# Use in pipeline
df = debug_features(df, add_domain_features(df))
```

### 4. Hyperparameter Grid Search (Alternative to Optuna)

```python
# Alternative to Optuna for simpler tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
}

grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

---

## 📊 Visualization & Analysis

### 1. Feature Importance

```python
# Add to train.py after model training
import matplotlib.pyplot as plt

# For tree-based models (RF, XGB, Cat, LGBM)
model = trained_models['RF']
importances = model.feature_importances_

plt.figure(figsize=(10, 8))
plt.barh(range(len(importances)), importances)
plt.title('Feature Importance')
plt.show()
```

### 2. Learning Curves

```python
# Detect overfitting/underfitting
from sklearn.model_selection import learning_curve

sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(sizes, train_scores.mean(axis=1), label='Train')
plt.plot(sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()
```

### 3. Prediction Error Analysis

```python
# Analyze prediction errors
from sklearn.metrics import mean_absolute_error, median_absolute_error

y_pred = model.predict(X_test)
errors = y_test - y_pred

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(errors, bins=50)
plt.title('Prediction Errors Distribution')
plt.xlabel('Error ($)')

plt.subplot(1, 2, 2)
plt.scatter(y_test, errors, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Errors vs Actual Price')
plt.xlabel('Actual Price')
plt.ylabel('Error ($)')

plt.show()

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Median AE: {median_absolute_error(y_test, y_pred):.2f}")
print(f"Error Std: {errors.std():.2f}")
```

---

## 🔄 Workflow Optimizations

### 1. Incremental Training

```python
# Train models one at a time instead of parallel
# Useful for debugging and monitoring memory

def train_sequentially(models, X_train, y_train, X_test, y_test, feature_pipe):
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        result = evaluate_single_model(
            name, model, X_train, y_train, X_test, y_test, feature_pipe
        )
        results.append(result)
        print(f"✅ {name} complete - RMSE: {result['test_rmse']:.2f}")
    
    return pd.DataFrame(results)
```

### 2. Cross-Validation Analysis

```python
# Detailed CV analysis
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    pipe, X_train, y_train,
    cv=5,
    scoring=['neg_rmse', 'r2'],
    return_train_score=True
)

# Analyze fold-wise performance
for fold in range(5):
    test_rmse = -cv_results[f'split{fold}_test_neg_rmse']
    print(f"Fold {fold}: {test_rmse:.2f}")

print(f"Mean RMSE: {-cv_results['test_neg_rmse'].mean():.2f}")
print(f"Std RMSE: {cv_results['test_neg_rmse'].std():.2f}")
```

### 3. Selective Feature Engineering

```python
# Enable/disable features as needed
from modules.feature_engineering import add_domain_features

def add_domain_features_selective(df, include_interactions=True):
    df = df.copy()
    
    # Core features
    df["TotalSF"] = df[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].fillna(0).sum(axis=1)
    
    # Optional interactions
    if include_interactions:
        df["Quality_Area_Interaction"] = df["OverallQual"] * df["GrLivArea"]
    
    return df
```

---

## 🐛 Debugging & Logging

### 1. Verbose Logging

```python
# Add to config.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### 2. Step-by-Step Debugging

```python
# In train.py, add debug prints
def main():
    print("Step 1: Loading data...")
    df = load_data(data_path)
    print(f"  ✅ Shape: {df.shape}")
    
    print("Step 2: Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"  ✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("Step 3: Creating pipeline...")
    feature_pipe = make_feature_space(X_train, X_test)
    print(f"  ✅ Pipeline created")
    
    print("Step 4: Training models...")
    # Continue...
```

### 3. Profile Code Performance

```python
# Profile specific functions
import cProfile
import pstats

def profile_training():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your training code here
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

# Run with: python -m cProfile train.py
```

---

## 🚀 Advanced Techniques

### 1. Ensemble Methods

```python
# Combine multiple models
from sklearn.ensemble import StackingRegressor

# Create ensemble
base_models = [
    ('rf', RandomForestRegressor(...)),
    ('ridge', Ridge(...)),
    ('xgb', xgb.XGBRegressor(...)),
]

ensemble = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge()
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
```

### 2. Custom Scoring Metrics

```python
# Define custom metrics
from sklearn.metrics import make_scorer

def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error"""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# Use in cross-validation
scorer = make_scorer(rmsle, greater_is_better=False)
scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
```

### 3. Automated Feature Selection

```python
# Select important features
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(f_regression, k=50)
X_selected = selector.fit_transform(X_train, y_train)

selected_features = X_train.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")
```

---

## 📦 Integration Tips

### 1. Use in FastAPI/Flask

```python
# inference_api.py
from fastapi import FastAPI
from modules.cache_manager import MLflowTrainingCacheManager

app = FastAPI()
cache = MLflowTrainingCacheManager()

# Load model once at startup
model, _ = cache.load_results('final_model')

@app.post("/predict")
def predict(features: dict):
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    return {"price": prediction}
```

### 2. Batch Prediction

```python
# Predict on large dataset efficiently
def batch_predict(model, df, batch_size=1000):
    predictions = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        pred = model.predict(batch)
        predictions.extend(pred)
    return np.array(predictions)
```

### 3. Model Versioning

```python
# Save different model versions
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/model_{timestamp}.pkl"

cache.save_results(
    model,
    'final_model',
    metadata={
        'version': '1.0',
        'timestamp': timestamp,
        'rmse': 20500.45,
        'r2': 0.8934,
        'features': feature_list
    }
)
```

---

## 🎓 Learning Resources

### Study the Code:
1. Start with `train.py` - understand the main flow
2. Review `modules/feature_engineering.py` - see how features are created
3. Check `modules/transformers.py` - understand data transformations
4. Explore `modules/models.py` - see model training logic

### Experiment:
1. Create a test notebook with small data
2. Modify hyperparameters and observe results
3. Add new features and measure impact
4. Try different model combinations

### Best Practices:
- Always use version control (git)
- Document your changes
- Cache intermediate results
- Monitor memory and time
- A/B test changes carefully

---

**Happy Development! 🚀**
