"""
Main pipeline orchestration script.
Coordinates the entire ML pipeline from data loading to model training and saving.
Run this script to execute the complete pipeline: python run_pipeline.py
"""

import logging
import sys
from pathlib import Path
from typing import Tuple
import dill
import joblib

import mlflow
import pandas as pd

from src.core.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    LOGS_DIR,
    RANDOM_STATE,
    EXPLAINER_DIR
)
from src.core.utils import Logger, MLflowTrainingCacheManager
from src.ml_pipeline.data_loader import prepare_data
from src.ml_pipeline.preprocessing import make_feature_space, save_scalers
from src.ml_pipeline.trainer import (
    build_base_models,
    run_baseline_evaluation,
    run_hyperparameter_tuning
)
from src.ml_pipeline.evaluation import get_metrics
from src.ml_pipeline.explainability import ModelExplainer


# Setup logging
logger = Logger(__name__, LOGS_DIR / "pipeline.log")


def setup_mlflow() -> None:
    """Configure MLflow tracking"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"[OK] MLflow tracking configured: {MLFLOW_TRACKING_URI}")


def run_pipeline(
    download_data: bool = True,
    run_baseline: bool = True,
    run_tuning: bool = True,
    n_tuning_trials: int = 15,
    save_models: bool = True
) -> bool:
    """
    Execute complete ML pipeline.
    
    Parameters:
        download_data: Whether to download dataset if missing
        run_baseline: Whether to run baseline model evaluation
        run_tuning: Whether to run hyperparameter tuning
        n_tuning_trials: Number of Optuna trials for tuning
        save_models: Whether to save trained models
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("=" * 80)
        logger.info("[START] STARTING HOUSE PRICE PREDICTION PIPELINE")
        logger.info("=" * 80)
        
        # ====================================================================
        # STEP 1: DATA LOADING
        # ====================================================================
        logger.info("\n[DATA] STEP 1: Data Loading")
        logger.info("-" * 80)
        
        with mlflow.start_run(run_name="Pipeline_Execution", nested=False):
            X_train, X_test, y_train, y_test = prepare_data(
                download_if_missing=download_data
            )
            logger.info(f"[OK] Data loaded: train={X_train.shape}, test={X_test.shape}")
            
            # ====================================================================
            # STEP 2: FEATURE ENGINEERING
            # ====================================================================
            logger.info("\n[FEATURE] STEP 2: Feature Engineering")
            logger.info("-" * 80)
            
            with mlflow.start_run(run_name="Feature_Engineering", nested=True):
                feature_pipe = make_feature_space(X_train, X_test)
                logger.info("[OK] Feature pipeline created")
                
                # Fit preprocessor
                feature_pipe.fit(X_train, y_train)
                logger.info("[OK] Feature pipeline fitted")
                
                # Save preprocessor for inference
                if save_models:
                    save_scalers(feature_pipe)
            
            # ====================================================================
            # STEP 3: BASELINE MODEL EVALUATION
            # ====================================================================
            if run_baseline:
                logger.info("\n[MODEL] STEP 3: Baseline Model Evaluation")
                logger.info("-" * 80)
                
                with mlflow.start_run(run_name="Baseline_Models", nested=True):
                    models = build_base_models()
                    
                    baseline_df, preds, top5 = run_baseline_evaluation(
                        models,
                        X_train, y_train,
                        X_test, y_test,
                        feature_pipe,
                        n_jobs=1  # Use 1 job to avoid memory issues
                    )
                    
                    if baseline_df.empty:
                        logger.error("[FAIL] Baseline evaluation failed")
                        return False
                    
                    logger.info(f"\n[RESULTS] Baseline Results:")
                    logger.info(baseline_df.to_string())
                    
                    # Save baseline results
                    baseline_df.to_csv(LOGS_DIR / "baseline_results.csv", index=False)
                    logger.info(f"[OK] Baseline results saved to {LOGS_DIR / 'baseline_results.csv'}")
            else:
                logger.info("\n[SKIP] Skipping Baseline Model Evaluation")
                top5 = ["XGB", "Cat", "LGBM", "RF", "Ridge"]  # Default top models
            
            # ====================================================================
            # STEP 4: HYPERPARAMETER TUNING
            # ====================================================================
            if run_tuning:
                logger.info("\n[TUNE] STEP 4: Hyperparameter Tuning")
                logger.info("-" * 80)
                
                with mlflow.start_run(run_name="Tuning", nested=True):
                    tuned, histories = run_hyperparameter_tuning(
                        top5,
                        X_train, y_train,
                        feature_pipe,
                        n_trials=n_tuning_trials
                    )
                    
                    if not tuned:
                        logger.warning("[WARN] No models were tuned successfully")
                        tuned = {}
            else:
                logger.info("\n[SKIP] Skipping Hyperparameter Tuning")
                tuned = {}
            
            # ====================================================================
            # STEP 5: FINAL MODEL SELECTION AND EVALUATION
            # ====================================================================
            logger.info("\n[FINAL] STEP 5: Final Model Selection")
            logger.info("-" * 80)
            
            with mlflow.start_run(run_name="Final_Model", nested=True):
                # Select best model (from tuned models or baseline)
                if tuned:
                    best_model_name = min(tuned.keys(), 
                                         key=lambda k: tuned[k]['best_cv_rmse'])
                    best_estimator = tuned[best_model_name]['estimator']
                    logger.info(f"[SELECTED] Best tuned model: {best_model_name}")
                else:
                    if run_baseline:
                        best_model_name = baseline_df.iloc[0]['model']
                        logger.info(f"[SELECTED] Best baseline model: {best_model_name}")
                    else:
                        logger.error("[FAIL] No baseline results available")
                        return False
                
                # Build final pipeline
                from sklearn.pipeline import Pipeline
                from sklearn.base import clone
                
                models = build_base_models()
                if best_model_name not in models:
                    logger.error(f"Model {best_model_name} not in base models")
                    return False
                
                final_model = clone(models[best_model_name])
                if best_model_name in tuned:
                    best_params = tuned[best_model_name]['best_params']
                    final_model.set_params(
                        **{k: v for k, v in best_params.items() 
                          if k not in ['n_jobs', 'random_state']}
                    )
                
                final_pipe = Pipeline([
                    ("features", clone(feature_pipe)),
                    ("model", final_model)
                ])
                
                # Fit on full training data
                final_pipe.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = final_pipe.predict(X_test)
                metrics = get_metrics(y_test, y_pred)
                
                logger.info(f"\n[METRICS] Final Model Performance:")
                logger.info(f"   RMSE: {metrics['rmse']:.4f}")
                logger.info(f"   MAE:  {metrics['mae']:.4f}")
                logger.info(f"   R2:   {metrics['r2']:.4f}")
                
                # Log final metrics to MLflow
                mlflow.log_metrics({
                    "final_rmse": metrics['rmse'],
                    "final_mae": metrics['mae'],
                    "final_r2": metrics['r2']
                })
                mlflow.log_param("final_model", best_model_name)
            
            # ====================================================================
            # STEP 6: SAVE ARTIFACTS
            # ====================================================================
            if save_models:
                logger.info("\n[SAVE] STEP 6: Saving Artifacts")
                logger.info("-" * 80)
                
                try:
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    
                    # Save model with dill (handles custom classes better)
                    model_path = MODELS_DIR / f"final_model_{best_model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        dill.dump(final_pipe, f)
                    logger.info(f"[OK] Model saved: {model_path}")
                    
                    # Also save with joblib for compatibility
                    model_path_joblib = MODELS_DIR / f"final_model_{best_model_name}.joblib"
                    joblib.dump(final_pipe, model_path_joblib)
                    logger.info(f"[OK] Model saved (joblib): {model_path_joblib}")
                    
                    # Log artifacts to MLflow
                    try:
                        mlflow.log_artifact(str(model_path), "models")
                        mlflow.log_artifact(str(MODELS_DIR / "preprocessor.joblib"), "models")
                        logger.info("[OK] Models logged to MLflow")
                    except Exception as e:
                        logger.warning(f"[WARN] Failed to log models to MLflow: {e}")
                    
                except Exception as e:
                    logger.error(f"[FAIL] Failed to save models: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # ====================================================================
                # CREATE AND SAVE SHAP EXPLAINER (SEPARATE from model save)
                # ====================================================================
                logger.info("\n[XAI] STEP 7: Creating SHAP Explainer for interpretability")
                logger.info("-" * 80)
                
                try:
                    # Extract feature transformer and model from pipeline
                    feature_transformer = final_pipe.named_steps['features']
                    model_estimator = final_pipe.named_steps['model']
                    
                    logger.info(f"[DEBUG] Feature transformer type: {type(feature_transformer).__name__}")
                    logger.info(f"[DEBUG] Model type: {type(model_estimator).__name__}")
                    logger.info(f"[DEBUG] X_train shape: {X_train.shape}")
                    
                    # Transform training data
                    logger.info("[DEBUG] Transforming training data...")
                    X_train_transformed = feature_transformer.transform(X_train)
                    logger.info(f"[OK] Data transformed. Shape: {X_train_transformed.shape}")
                    
                    # Ensure it's a proper array
                    if hasattr(X_train_transformed, 'toarray'):  # sparse matrix
                        X_train_transformed = X_train_transformed.toarray()
                        logger.info(f"[DEBUG] Converted sparse matrix to dense array")
                    
                    # Create and fit explainer
                    logger.info("[DEBUG] Creating ModelExplainer instance...")
                    explainer = ModelExplainer(model_estimator, feature_transformer)
                    
                    logger.info("[DEBUG] Fitting explainer with training data...")
                    explainer.fit(X_train_transformed, y_train)
                    logger.info("[OK] Explainer fitted successfully")
                    
                    # Save explainer
                    logger.info("[DEBUG] Saving explainer to disk...")
                    explainer_path = explainer.save()
                    logger.info(f"[OK] SHAP explainer saved: {explainer_path}")
                    
                    # Verify it was saved
                    if explainer_path.exists():
                        logger.info(f"[VERIFY] Explainer file exists, size: {explainer_path.stat().st_size} bytes")
                    else:
                        logger.error(f"[ERROR] Explainer file not found after save: {explainer_path}")
                    
                    # Log explainer to MLflow
                    try:
                        mlflow.log_artifact(str(explainer_path), "explainers")
                        logger.info("[OK] Explainer logged to MLflow")
                    except Exception as e:
                        logger.warning(f"[WARN] Failed to log explainer to MLflow: {e}")
                    
                except ImportError as e:
                    logger.error(f"[ERROR] SHAP library not available: {e}")
                    logger.error("Please install SHAP: pip install shap")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to create SHAP explainer: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.warning("XAI features will be unavailable in the API")
        
        # ====================================================================
        # COMPLETION
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Models saved to: {MODELS_DIR}")
        logger.info(f"Logs saved to: {LOGS_DIR}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Setup
    setup_mlflow()
    
    # Run pipeline with options
    success = run_pipeline(
        download_data=True,
        run_baseline=True,
        run_tuning=False,  # Set to True for full pipeline (takes longer)
        n_tuning_trials=15,
        save_models=True
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
