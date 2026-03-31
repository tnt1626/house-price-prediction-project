"""
Evaluation module for model performance assessment.
Contains metrics calculation and logging functions.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow

from src.core.utils import Logger


logger = Logger(__name__)


def get_scorers() -> Dict[str, str]:
    """
    Get scikit-learn scorer strings for cross-validation.
    Uses string identifiers to ensure compatibility across sklearn versions.
    
    Returns:
        Dictionary of scorer names to scorer identifiers
    """
    return {
        'neg_rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Parameters:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary with RMSE, MAE, and R² scores
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mse": mse
    }


def log_metrics_to_mlflow(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> None:
    """
    Log metrics to MLflow.
    
    Parameters:
        metrics: Metrics dictionary
        y_true: True values (for additional stats)
        y_pred: Predicted values
        prefix: Prefix for metric names
    """
    if mlflow.active_run():
        for metric_name, value in metrics.items():
            full_name = f"{prefix}_{metric_name}" if prefix else metric_name
            mlflow.log_metric(full_name, float(value))


def calculate_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate residual statistics.
    
    Parameters:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with residual statistics
    """
    residuals = y_true - y_pred
    
    return {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "min_residual": float(np.min(residuals)),
        "max_residual": float(np.max(residuals)),
        "q25_residual": float(np.percentile(residuals, 25)),
        "median_residual": float(np.median(residuals)),
        "q75_residual": float(np.percentile(residuals, 75))
    }


def generate_predictions_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Generate detailed predictions report.
    
    Parameters:
        y_test: True values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        DataFrame with prediction details
    """
    report_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Residual': y_test - y_pred,
        'Absolute_Error': np.abs(y_test - y_pred),
        'Percent_Error': np.abs((y_test - y_pred) / y_test * 100)
    })
    
    return report_df


def get_model_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, any]:
    """
    Generate comprehensive model evaluation report.
    
    Parameters:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Dictionary with complete evaluation report
    """
    metrics = get_metrics(y_true, y_pred)
    residuals = calculate_residuals(y_true, y_pred)
    
    report = {
        "model_name": model_name,
        "metrics": metrics,
        "residuals": residuals,
        "predictions_count": len(y_true),
        "underprediction_count": int(np.sum(y_pred < y_true)),
        "overprediction_count": int(np.sum(y_pred > y_true))
    }
    
    return report
