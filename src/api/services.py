"""
Service layer for business logic.
Handles model loading, preprocessing, and predictions.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging

from src.core.config import MODELS_DIR, SCALERS_DIR
from src.api.schemas import HousePriceInput, PredictionResponse

 
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making price predictions."""
    
    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """
        Initialize prediction service.
        
        Parameters:
            model_path: Path to trained model
            scaler_path: Path to preprocessing scaler
        """
        self.model = None
        self.preprocessor = None
        self.model_name = "None"
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_preprocessor(scaler_path)
    
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
    
    def is_ready(self) -> bool:
        """Check if service is ready for predictions."""
        return self.model is not None
    
    def input_to_dataframe(self, data: HousePriceInput) -> pd.DataFrame:
        """
        Convert input schema to DataFrame.
        
        Parameters:
            data: Input data
            
        Returns:
            DataFrame with feature values
        """
        # Convert Pydantic model to dict and then to DataFrame
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])
        return df
    
    def predict_single(self, data: HousePriceInput) -> Dict[str, Any]:
        """
        Make prediction for a single house.
        
        Parameters:
            data: Input house data
            
        Returns:
            Dictionary with prediction and metadata
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Please load model first.")
        
        try:
            # Convert input to DataFrame
            df = self.input_to_dataframe(data)
            
            # Preprocess if available
            if self.preprocessor:
                try:
                    df_processed = self.preprocessor.transform(df)
                except Exception as e:
                    logger.warning(f"Preprocessing failed: {e}, using raw features")
                    df_processed = df
            else:
                df_processed = df
            
            # Make prediction
            prediction = self.model.predict(df_processed)[0]
            
            # Calculate rough confidence (0-1 scale based on prediction uncertainty)
            # This is a simplified approach; more sophisticated methods could be used
            confidence = 0.85  # Default confidence
            
            return {
                "predicted_price": float(prediction),
                "confidence": confidence,
                "model_name": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
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
        for data in data_list:
            try:
                pred = self.predict_single(data)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict for house {len(predictions)}: {e}")
                predictions.append({
                    "predicted_price": None,
                    "confidence": 0.0,
                    "model_name": self.model_name,
                    "error": str(e)
                })
        
        return predictions


class ModelRegistry:
    """Simple registry for managing models."""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.scalers_dir = SCALERS_DIR
    
    def list_available_models(self) -> list:
        """List all available trained models."""
        if not self.models_dir.exists():
            return []
        return [f.stem for f in self.models_dir.glob("*.pkl") 
                if f.is_file() and f.stem != "preprocessor"]
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path to a model."""
        return self.models_dir / f"{model_name}.pkl"
    
    def get_scaler_path(self) -> Path:
        """Get path to preprocessing scaler."""
        return self.scalers_dir / "preprocessor.joblib"
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model exists."""
        return self.get_model_path(model_name).exists()
    
    def scaler_exists(self) -> bool:
        """Check if scaler exists."""
        return self.get_scaler_path().exists()


def create_default_service() -> PredictionService:
    """
    Create a prediction service with default models.
    
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
    return service
