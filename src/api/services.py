"""
Service layer for business logic.
Handles model loading, preprocessing, and predictions, including SHAP-based explanations.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging

from src.core.config import MODELS_DIR, SCALERS_DIR, EXPLAINER_DIR, SCHEMA_TO_DATA_MAPPING
from src.api.schemas import HousePriceInput, PredictionResponse
from src.ml_pipeline.explainability import ModelExplainer
from sklearn.pipeline import Pipeline as SklearnPipeline

 
logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing model loading and predictions."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize model service.
        
        Parameters:
            model_path: Path to trained model
        """
        self.model = None
        self.model_name = "None"
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> bool:
        """
        Load trained model from disk.
        
        Parameters:
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            self.model_name = model_path.stem
            logger.info(f"[OK] Model loaded: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def predict(self, X: pd.DataFrame) -> float:
        """
        Make a single prediction.
        
        Parameters:
            X: Feature data as DataFrame
            
        Returns:
            Predicted price
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded")
        
        try:
            prediction = self.model.predict(X)
            return float(prediction[0])
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for multiple samples.
        
        Parameters:
            X: Feature data as DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded")
        
        return self.model.predict(X)


class PreprocessingService:
    """Service for data preprocessing and transformation."""
    
    def __init__(self, scaler_path: Optional[Path] = None):
        """
        Initialize preprocessing service.
        
        Parameters:
            scaler_path: Path to preprocessing scaler
        """
        self.preprocessor = None
        
        if scaler_path:
            self.load_preprocessor(scaler_path)
    
    def load_preprocessor(self, scaler_path: Path) -> bool:
        """
        Load preprocessing pipeline from disk.
        
        Parameters:
            scaler_path: Path to preprocessor file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            scaler_path = Path(scaler_path)
            if not scaler_path.exists():
                logger.error(f"Scaler file not found: {scaler_path}")
                return False
            
            self.preprocessor = joblib.load(scaler_path)
            logger.info(f"[OK] Preprocessor loaded: {scaler_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            return False
    
    def input_to_dataframe(self, data: HousePriceInput) -> pd.DataFrame:
        """
        Convert input schema to DataFrame.
        Renames columns from schema names to data column names if needed.
        Ensures all required columns exist with proper defaults.
        
        Parameters:
            data: Input data
            
        Returns:
            DataFrame with feature values (using actual data column names)
        """
        # Get all field names from the schema
        schema_fields = set(HousePriceInput.model_fields.keys())
        
        # Convert to dict, including all fields with exclude_unset=False
        data_dict = data.model_dump(exclude_unset=False)
        
        # Apply column name mapping from schema to data column names
        mapped_dict = {}
        for k, v in data_dict.items():
            # Get the mapped column name from SCHEMA_TO_DATA_MAPPING, or use original if not mapped
            actual_col_name = SCHEMA_TO_DATA_MAPPING.get(k, k)
            mapped_dict[actual_col_name] = v
        
        # Also add columns using schema names as fallback
        for field_name in schema_fields:
            if field_name not in mapped_dict:
                actual_col_name = SCHEMA_TO_DATA_MAPPING.get(field_name, field_name)
                if actual_col_name not in mapped_dict:
                    mapped_dict[actual_col_name] = data_dict.get(field_name, None)
        
        # Create DataFrame with a single row
        df = pd.DataFrame([mapped_dict])
        
        # Ensure no duplicate columns, remove fully NaN columns to avoid issues
        df = df.loc[:, ~df.columns.duplicated()]
        
        logger.debug(f"[DEBUG] Input DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
        return df
    
    def _ensure_preprocessor_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all columns expected by preprocessor are present in DataFrame.
        This handles the case where preprocessor was trained on specific columns
        but prediction input may have missing columns.
        
        Parameters:
            df: Input DataFrame
            
        Returns:
            DataFrame with all required columns
        """
        if self.preprocessor is None:
            return df
        
        try:
            # Try to get feature names from preprocessor
            original_columns = None
            
            # Get all input feature names expected by preprocessor
            if hasattr(self.preprocessor, 'named_steps'):
                # Get the add_domain step's output columns if available
                if 'add_domain' in self.preprocessor.named_steps:
                    # The add_domain step doesn't change column names drastically
                    pass
                
                # Get the preprocessing step's column names
                if 'preproc' in self.preprocessor.named_steps:
                    preproc = self.preprocessor.named_steps['preproc']
                    if hasattr(preproc, 'named_steps'):
                        if 'ct' in preproc.named_steps:
                            ct = preproc.named_steps['ct']
                            if hasattr(ct, 'transformers_'):
                                # List all columns that ColumnTransformer expects
                                all_transformer_cols = []
                                for name, transformer, cols in ct.transformers_:
                                    all_transformer_cols.extend(cols)
                                original_columns = all_transformer_cols
            
            if original_columns:
                # Add missing columns with default values
                for col in original_columns:
                    if col not in df.columns:
                        # Infer type and add appropriate default
                        df[col] = 0  # Default to 0 for numeric columns
                        logger.debug(f"[DEBUG] Added missing column '{col}' with default value 0")
        
        except Exception as e:
            logger.warning(f"Could not ensure preprocessor columns: {e}")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using preprocessor.
        
        Parameters:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if self.preprocessor is None:
            return df
        
        try:
            X_transformed = self.preprocessor.transform(df)
            
            # Ensure transformed data is DataFrame with column names
            if not isinstance(X_transformed, pd.DataFrame):
                try:
                    cols = self.preprocessor.get_feature_names_out()
                except Exception:
                    cols = [f"f{i}" for i in range(X_transformed.shape[1])]
                
                X_transformed = pd.DataFrame(X_transformed, columns=cols)
            
            return X_transformed
        except Exception as e:
            logger.error(f"Transformation error: {e}")
            raise


class ExplanationService:
    """Service for SHAP-based model explanations."""
    
    def __init__(self):
        """Initialize explanation service."""
        self.explainer = None
    
    def load_explainer(
        self,
        model: Any,
        preprocessor: Any,
        explainer_path: Optional[Path] = None
    ) -> bool:
        """
        Load SHAP explainer from disk.
        
        Parameters:
            model: Trained model
            preprocessor: Preprocessing pipeline
            explainer_path: Path to explainer file (default: EXPLAINER_DIR/shap_explainer.joblib)
            
        Returns:
            True if successful, False otherwise
        """
        if model is None or preprocessor is None:
            logger.error("Model and preprocessor must be provided to load explainer")
            return False
        
        if explainer_path is None:
            explainer_path = EXPLAINER_DIR / "shap_explainer.joblib"
        
        try:
            explainer_path = Path(explainer_path)
            if not explainer_path.exists():
                logger.warning(f"Explainer file not found: {explainer_path}")
                return False
            
            # Get original feature names from schema
            original_feature_names = list(HousePriceInput.model_fields.keys())
            
            self.explainer = ModelExplainer.load(
                model,
                preprocessor,
                explainer_path,
                original_feature_names=original_feature_names
            )
            logger.info(f"[OK] Explainer loaded: {explainer_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load explainer: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if explainer is loaded."""
        return self.explainer is not None
    
    def explain(
        self,
        X_processed: pd.DataFrame,
        original_input: Dict[str, Any],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate local explanation for a sample.
        
        Parameters:
            X_processed: Processed feature data as DataFrame
            original_input: Original input data
            top_k: Number of top features to return
            
        Returns:
            Explanation data with base value and feature contributions
        """
        if not self.is_ready():
            raise RuntimeError("Explainer not loaded")
        
        try:
            explanation = self.explainer.get_local_explanation(
                X_single=X_processed,
                original_input=original_input,
                top_k=top_k
            )
            return explanation
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            raise


class PredictionService:
    """Orchestrator service for making price predictions with preprocessing and explanations."""
    
    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """
        Initialize prediction service.
        
        Parameters:
            model_path: Path to trained model
            scaler_path: Path to preprocessing scaler
        """
        self.model_service = ModelService(model_path=model_path)
        self.preprocessing_service = PreprocessingService(scaler_path=scaler_path)
        self.explanation_service = ExplanationService()
    
    @property
    def model_name(self) -> str:
        """Get loaded model name."""
        return self.model_service.model_name
    
    def load_model(self, model_path: Path) -> bool:
        """Load trained model from disk."""
        return self.model_service.load_model(model_path)
    
    def load_preprocessor(self, scaler_path: Path) -> bool:
        """Load preprocessing pipeline from disk."""
        return self.preprocessing_service.load_preprocessor(scaler_path)
    
    def is_ready(self) -> bool:
        """Check if service is ready for predictions."""
        return self.model_service.is_ready()
    def input_to_dataframe(self, data: HousePriceInput) -> pd.DataFrame:
        """Convert input schema to DataFrame."""
        return self.preprocessing_service.input_to_dataframe(data)
    
    def predict_single(self, input_data: HousePriceInput) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Parameters:
            input_data: Input house data
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Please check model path.")
        
        try:
            # Step 1: Convert schema to DataFrame
            df = self.preprocessing_service.input_to_dataframe(input_data)
            
            # Step 2: Ensure all required columns exist
            df = self.preprocessing_service._ensure_preprocessor_columns(df)
            
            # Step 3: Determine if we need to transform data
            if isinstance(self.model_service.model, SklearnPipeline):
                # Pipeline handles preprocessing internally
                logger.info("Model is a full Pipeline. Predicting directly from raw features.")
                X_input = df
            else:
                # Transform data manually
                if self.preprocessing_service.preprocessor:
                    X_input = self.preprocessing_service.transform(df)
                else:
                    logger.warning("No preprocessor loaded, using raw features")
                    X_input = df
            
            # Step 4: Make prediction
            prediction = self.model_service.predict(X_input)
            
            return {
                "predicted_price": prediction,
                "confidence": 0.85,
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, data_list: list) -> list:
        """
        Make predictions for multiple houses.
        
        Parameters:
            data_list: List of HousePriceInput objects
            
        Returns:
            List of predictions
        """
        predictions = []
        for idx, data in enumerate(data_list):
            try:
                pred = self.predict_single(data)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict for house {idx}: {e}")
                predictions.append({
                    "predicted_price": None,
                    "confidence": 0.0,
                    "model_name": self.model_name,
                    "error": str(e)
                })
        
        return predictions
    
    def load_explainer(self, explainer_path: Optional[Path] = None) -> bool:
        """Load SHAP explainer from disk."""
        return self.explanation_service.load_explainer(
            self.model_service.model,
            self.preprocessing_service.preprocessor,
            explainer_path
        )
    
    def is_explainer_ready(self) -> bool:
        """Check if SHAP explainer is ready for explanations."""
        return self.explanation_service.is_ready()
    
    def predict_and_explain(
        self,
        data: HousePriceInput,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Make a prediction and provide SHAP-based explanation.
        
        Parameters:
            data: Input house data
            top_k: Number of top features to explain
            
        Returns:
            Dictionary with prediction and explanation
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Please check model path.")
        
        if not self.explanation_service.is_ready():
            raise RuntimeError("Explainer not loaded. Please check explainer path.")
        
        try:
            # Step 1: Convert schema to DataFrame
            df = self.preprocessing_service.input_to_dataframe(data)
            original_input = data.model_dump()
            
            # Step 2: Determine if we need to transform data
            if isinstance(self.model_service.model, SklearnPipeline):
                logger.info("Model is a full Pipeline. Using raw features for prediction.")
                X_input = df
                
                # Extract processed features for SHAP
                try:
                    preprocessor_steps = self.model_service.model.named_steps
                    if 'preprocessor' in preprocessor_steps:
                        df_processed = preprocessor_steps['preprocessor'].transform(df)
                    else:
                        if self.preprocessing_service.preprocessor:
                            df_processed = self.preprocessing_service.transform(df)
                        else:
                            df_processed = df
                except Exception:
                    if self.preprocessing_service.preprocessor:
                        df_processed = self.preprocessing_service.transform(df)
                    else:
                        df_processed = df
            else:
                # Transform data manually
                if self.preprocessing_service.preprocessor:
                    df_processed = self.preprocessing_service.transform(df)
                else:
                    df_processed = df
                X_input = df_processed
            
            # Ensure processed data is a DataFrame
            if not isinstance(df_processed, pd.DataFrame):
                try:
                    cols = self.preprocessing_service.preprocessor.get_feature_names_out()
                except Exception:
                    cols = [f"f{i}" for i in range(df_processed.shape[1])]
                
                df_processed = pd.DataFrame(df_processed, columns=cols)
            
            # Step 3: Make prediction
            prediction = self.model_service.predict(X_input)
            
            # Step 4: Get explanation
            explanation_data = self.explanation_service.explain(
                df_processed,
                original_input,
                top_k=top_k
            )
            
            return {
                "predicted_price": prediction,
                "confidence": 0.85,
                "model_name": self.model_name,
                "base_value": explanation_data["base_value"],
                "explanations": explanation_data["explanations"]
            }
            
        except Exception as e:
            logger.error(f"Prediction with explanation failed: {e}")
            raise


class ModelRegistry:
    """Simple registry for managing models."""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.scalers_dir = SCALERS_DIR
    
    def list_available_models(self) -> list:
        """List all available trained models (supports .pkl and .joblib formats)."""
        if not self.models_dir.exists():
            return []
        
        models = []
        # Tìm cả .pkl và .joblib files
        for f in self.models_dir.glob("*"):
            if f.is_file() and f.stem != "preprocessor":
                if f.suffix in [".pkl", ".joblib"]:
                    models.append(f.stem)
        
        return list(set(models))  # Remove duplicates
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path to a model (tries .joblib first, then .pkl)."""
        # Try .joblib first
        joblib_path = self.models_dir / f"{model_name}.joblib"
        if joblib_path.exists():
            return joblib_path
        
        # Fallback to .pkl
        pkl_path = self.models_dir / f"{model_name}.pkl"
        if pkl_path.exists():
            return pkl_path
        
        # Default to .joblib if neither exists (for error clarity)
        return joblib_path
    
    def get_scaler_path(self) -> Path:
        """Get path to preprocessing scaler."""
        return self.scalers_dir / "preprocessor.joblib"
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model exists."""
        path = self.get_model_path(model_name)
        return path.exists()
    
    def scaler_exists(self) -> bool:
        """Check if scaler exists."""
        return self.get_scaler_path().exists()


def create_default_service() -> PredictionService:
    """
    Create a prediction service with default models and explainer.
    
    Returns:
        Configured PredictionService
    """
    registry = ModelRegistry()
    
    # Try to load the latest model
    models = registry.list_available_models()
    if not models:
        logger.warning("No models found in model registry")
        return PredictionService()
    
    # Use the first available model (sorted alphabetically)
    model_name = sorted(models)[0]
    model_path = registry.get_model_path(model_name)
    scaler_path = registry.get_scaler_path() if registry.scaler_exists() else None
    
    service = PredictionService(model_path=model_path, scaler_path=scaler_path)
    
    # Try to load explainer (non-blocking; continue even if it fails)
    if (EXPLAINER_DIR / "shap_explainer.joblib").exists():
        if not service.load_explainer():
            logger.warning("Could not load SHAP explainer, XAI features will be unavailable")
    
    return service