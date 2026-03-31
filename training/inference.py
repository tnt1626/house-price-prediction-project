#!/usr/bin/env python3
# ============================================================================
# Inference Script - Make predictions with trained model
# ============================================================================
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from modules.config import LOGS_DIR


def load_model(model_path):
    """Load a trained model from pickle file"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        try:
            import dill
            model = dill.load(f)
        except ImportError:
            print("⚠️  dill not available, trying standard pickle...")
            model = pickle.load(f)
    
    print(f"✅ Model loaded from: {model_path}")
    return model


def predict_prices(model, X_input):
    """Make price predictions"""
    predictions = model.predict(X_input)
    return predictions


def main(data_path, model_path=None):
    """Main inference pipeline"""
    
    print("=" * 70)
    print("🏠 HOUSE PRICE PREDICTION - INFERENCE")
    print("=" * 70)
    
    # Load data
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"\n📂 Loading data: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded: {df.shape}")
    
    # Handle target column if present
    if 'SalePrice' in df.columns:
        y_true = df['SalePrice'].values
        X = df.drop(['Id', 'SalePrice'], axis=1, errors='ignore')
        has_target = True
    else:
        y_true = None
        X = df.drop('Id', axis=1, errors='ignore')
        has_target = False
    
    print(f"✅ Features prepared: {X.shape}")
    
    # Load model
    if model_path is None:
        model_path = LOGS_DIR / "final_house_price_model.pkl"
    
    print(f"\n🤖 Loading model...")
    model = load_model(str(model_path))
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    y_pred = predict_prices(model, X)
    print(f"✅ Predictions generated: {len(y_pred)} samples")
    
    # Create results dataframe
    results = pd.DataFrame({
        'prediction': y_pred,
    })
    
    if has_target:
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        results['actual'] = y_true
        results['error'] = np.abs(y_true - y_pred)
        
        print(f"\n📊 Evaluation Metrics:")
        print(f"   RMSE: ${rmse:,.2f}")
        print(f"   R²: {r2:.4f}")
        print(f"   Mean Absolute Error: ${results['error'].mean():,.2f}")
    
    # Save results
    output_file = LOGS_DIR / "predictions.csv"
    results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ INFERENCE COMPLETE!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <data_path> [model_path]")
        print("\nExample:")
        print("  python inference.py ../data/test.csv")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = main(data_path, model_path)
