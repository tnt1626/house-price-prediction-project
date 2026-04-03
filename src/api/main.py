"""
FastAPI application for house price prediction.
Main entry point for the API service.
"""

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import API_HOST, API_PORT, EXPLAINER_DIR
from src.core.utils import Logger
from src.api.schemas import (
    HousePriceInput,
    PredictionResponse,
    BatchPredictionInput,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    PredictionWithExplainResponse
)
from src.api.services import PredictionService, create_default_service, ModelRegistry


# Global service instance
prediction_service: PredictionService = None
logger = Logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    global prediction_service
    logger.info("[START] Starting API server...")
    try:
        prediction_service = create_default_service()
        if not prediction_service.is_ready():
            logger.warning("[WARN] No model loaded - predictions will fail")
        else:
            logger.info("[OK] Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
    
    yield
    
    # Shutdown
    logger.info("[STOP] Shutting down API server...")


# Create FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root() -> dict:
    """Root endpoint"""
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="ok" if prediction_service and prediction_service.is_ready() else "degraded",
        model_loaded=prediction_service is not None and prediction_service.is_ready(),
        model_name=prediction_service.model_name if prediction_service else "None",
        version="1.0.0"
    )


# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(data: HousePriceInput) -> PredictionResponse:
    """
    Predict house price for a single property.
    
    - **LotArea**: Lot area in square feet
    - **OverallQual**: Overall quality rating (1-10)
    - **GrLivArea**: Above grade living area in square feet
    - ... other features
    
    Returns predicted sale price with confidence score.
    """
    if not prediction_service or not prediction_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready. Model not loaded."
        )
    
    try:
        result = prediction_service.predict_single(data)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(input_data: BatchPredictionInput) -> BatchPredictionResponse:
    """
    Predict house prices for multiple properties.
    
    Accepts a batch of up to 100 properties and returns predictions for each.
    """
    if not prediction_service or not prediction_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready. Model not loaded."
        )
    
    # Limit batch size
    if len(input_data.houses) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 100 properties"
        )
    
    try:
        predictions = prediction_service.predict_batch(input_data.houses)
        valid_predictions = [
            PredictionResponse(**p) for p in predictions 
            if "error" not in p
        ]
        
        return BatchPredictionResponse(
            predictions=valid_predictions,
            total_processed=len(input_data.houses)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict-explain", response_model=PredictionWithExplainResponse, tags=["Predictions"])
async def predict_with_explanation(data: HousePriceInput, top_features: int = 10) -> PredictionWithExplainResponse:
    """
    Predict house price with SHAP-based explainability.
    
    Returns predicted price along with impact analysis of top contributing features.
    Features are ranked by their SHAP values indicating positive or negative contribution to the price.
    
    **Parameters:**
    - **data**: House features input
    - **top_features**: Number of top contributing features to return (default: 10)
    
    **Response includes:**
    - **predicted_price**: ML model prediction
    - **base_value**: SHAP base value (expected/mean prediction)
    - **explanations**: List of features with SHAP values indicating individual contributions
    """
    if not prediction_service or not prediction_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready. Model not loaded."
        )
    
    if not prediction_service.is_explainer_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not loaded. XAI features are unavailable. Please retrain the model with explainer generation."
        )
    
    try:
        result = prediction_service.predict_and_explain(data, top_k=top_features)
        return PredictionWithExplainResponse(**result)
    except Exception as e:
        logger.error(f"Prediction with explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction with explanation failed: {str(e)}"
        )


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/models", tags=["Models"])
async def list_models() -> dict:
    """List all available trained models"""
    registry = ModelRegistry()
    models = registry.list_available_models()
    
    return {
        "available_models": models,
        "current_model": prediction_service.model_name if prediction_service else "None",
        "model_count": len(models),
        "has_preprocessor": registry.scaler_exists()
    }


@app.post("/models/load/{model_name}", tags=["Models"])
async def load_model(model_name: str) -> dict:
    """Load a specific model"""
    global prediction_service
    
    registry = ModelRegistry()
    if not registry.model_exists(model_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    try:
        model_path = registry.get_model_path(model_name)
        scaler_path = registry.get_scaler_path() if registry.scaler_exists() else None
        
        prediction_service = PredictionService(model_path=model_path, scaler_path=scaler_path)
        
        if not prediction_service.is_ready():
            raise Exception("Failed to load model")
        
        # Load explainer after model and preprocessor are loaded
        explainer_path = EXPLAINER_DIR / "shap_explainer.joblib"
        if explainer_path.exists():
            if not prediction_service.load_explainer():
                logger.warning(f"Could not load explainer for model '{model_name}'")
            else:
                logger.info(f"Explainer loaded for model '{model_name}'")
        
        logger.info(f"Model '{model_name}' loaded successfully")
        return {
            "status": "ok",
            "message": f"Model '{model_name}' loaded successfully",
            "model_name": model_name
        }
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, detail="").dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ).dict()
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
