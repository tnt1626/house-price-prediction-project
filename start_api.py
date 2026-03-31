#!/usr/bin/env python
"""
API Server Startup Script
Launch the FastAPI server for house price predictions.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.utils import Logger
from src.api.main import app
import uvicorn


if __name__ == "__main__":
    logger = Logger(__name__)
    
    logger.info("=" * 80)
    logger.info("[START] STARTING HOUSE PRICE PREDICTION API")
    logger.info("=" * 80)
    
    try:
        logger.info("API Server starting...")
        logger.info("[INFO] Documentation will be available at: http://localhost:8000/docs")
        
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("\n[STOP] Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[FAIL] Failed to start server: {e}")
        sys.exit(1)
