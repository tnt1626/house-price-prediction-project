import pickle
import json
import mlflow
from pathlib import Path
from datetime import datetime
import threading


class MLflowTrainingCacheManager:
    """Enhanced caching system for training and evaluation results with MLflow integration"""

    def __init__(self, base_path="training_cache"):
        self.base_path = Path(base_path)
        self.cache_lock = threading.Lock()

    def _ensure_cache_dir(self):
        """Create cache directory only when needed"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Could not create cache directory: {e}")

    def save_results(self, results, filename, metadata=None):
        """Save training/evaluation results to cache with MLflow logging"""
        with self.cache_lock:
            try:
                self._ensure_cache_dir()

                cache_file = self.base_path / f"{filename}.pkl"
                meta_file = self.base_path / f"{filename}_meta.json"

                # Save results
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)

                # Save metadata
                if metadata is None:
                    metadata = {}
                metadata.update({
                    'timestamp': datetime.now().isoformat(),
                    'filename': filename
                })

                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Log to MLflow
                with mlflow.start_run(run_name=f"Cache_Save_{filename}", nested=True):
                    mlflow.log_param("cache_filename", filename)
                    mlflow.log_param("cache_timestamp", metadata['timestamp'])
                    mlflow.log_artifact(str(cache_file), "cache")
                    mlflow.log_artifact(str(meta_file), "cache_metadata")

                print(f"💾 Cached training results: {filename}")
                return True
            except Exception as e:
                print(f"❌ Cache save failed: {e}")
                return False

    def load_results(self, filename):
        """Load training/evaluation results from cache"""
        with self.cache_lock:
            try:
                cache_file = self.base_path / f"{filename}.pkl"
                meta_file = self.base_path / f"{filename}_meta.json"

                if not cache_file.exists():
                    return None, None

                # Load results from file pickle :3
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)

                # Load metadata
                metadata = None
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)

                # Log cache hit to MLflow
                with mlflow.start_run(run_name=f"Cache_Load_{filename}", nested=True):
                    mlflow.log_param("cache_filename", filename)
                    mlflow.log_param("cache_hit", True)
                    if metadata:
                        mlflow.log_param("cache_timestamp", metadata.get('timestamp', 'unknown'))

                print(f"📂 Loaded cached results: {filename}")
                return results, metadata
            except Exception as e:
                print(f"❌ Cache load failed: {e}")
                return None, None

    def results_exist(self, filename):
        """Check if cached results exist"""
        cache_file = self.base_path / f"{filename}.pkl"
        return cache_file.exists()
