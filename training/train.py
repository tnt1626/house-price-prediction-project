#!/usr/bin/env python3
# ============================================================================
# Main Training Script - Run this to train models locally
# ============================================================================
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import train_test_split

from modules.config import (
    RANDOM_STATE, CACHE_DIR, LOGS_DIR, BASE_PATH, MLFLOW_DIR, USE_MLFLOW
)
from modules.cache_manager import MLflowTrainingCacheManager
from modules.feature_engineering import make_feature_space
from modules.models import base_models_dict, run_baseline_evaluation, model_summary

# Try MLflow integration (optional)
try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("⚠️  MLflow not installed. Continuing without MLflow tracking...")


def setup_mlflow():
    """Setup MLflow tracking (optional)"""
    if USE_MLFLOW and HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
            mlflow.set_experiment("House_Price_Prediction_Local")
            print("✅ MLflow initialized")
            return True
        except Exception as e:
            print(f"⚠️  MLflow setup failed: {e}")
            return False
    return False


def load_data(data_path):
    """Load training data"""
    print(f"\n📂 Loading data from: {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ ERROR: Data file not found at {data_path}")
        print("   Please update the data_path in the script or ensure the CSV file exists.")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def prepare_data(df, test_size=0.2):
    """Prepare and split data"""
    print(f"\n🔄 Preparing data...")
    
    # Remove ID column
    df = df.drop('Id', axis=1, errors='ignore')
    
    # Split features and target
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"].astype(float)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    
    print(f"✅ Data split complete:")
    print(f"   Train set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def main():
    """Main training pipeline"""
    
    print("=" * 70)
    print("🏠 HOUSE PRICE PREDICTION - LOCAL TRAINING")
    print("=" * 70)
    
    # Setup
    use_mlflow = setup_mlflow()
    training_cache = MLflowTrainingCacheManager(CACHE_DIR)
    
    # Load data
    data_path = Path("../data/train-house-prices-advanced-regression-techniques.csv")
    if not data_path.exists():
        # Try alternative path
        data_path = Path("./data/train-house-prices-advanced-regression-techniques.csv")
    
    df = load_data(str(data_path))
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Feature engineering
    print(f"\n⚙️  Creating feature engineering pipeline...")
    feature_pipe = make_feature_space(X_train, X_test)
    print(f"✅ Feature pipeline ready")
    
    # Model training
    print(f"\n🤖 Training models...")
    models = base_models_dict()
    print(f"✅ {len(models)} models initialized: {list(models.keys())}")
    
    if use_mlflow:
        with mlflow.start_run(run_name="Baseline_Training"):
            baseline_df, preds, top5 = run_baseline_evaluation(
                models, X_train, y_train, X_test, y_test, feature_pipe, n_jobs=-1
            )
    else:
        baseline_df, preds, top5 = run_baseline_evaluation(
            models, X_train, y_train, X_test, y_test, feature_pipe, n_jobs=-1
        )
    
    # Display results
    model_summary(baseline_df)
    
    # Cache results
    cache_data = {
        'baseline_df': baseline_df,
        'preds': preds,
        'top5': top5,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_pipe': feature_pipe
    }
    
    training_cache.save_results(cache_data, 'baseline_results_local', metadata={
        'n_models': len(models),
        'best_model': baseline_df.iloc[0]['model'] if not baseline_df.empty else 'None'
    })
    
    # Save results to CSV
    output_file = LOGS_DIR / "baseline_results.csv"
    baseline_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ BASELINE TRAINING COMPLETE!")
    print("=" * 70)
    
    return baseline_df, preds, top5, X_train, X_test, y_train, y_test, feature_pipe


if __name__ == "__main__":
    baseline_df, preds, top5, X_train, X_test, y_train, y_test, feature_pipe = main()
