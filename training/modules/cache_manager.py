# ============================================================================
# Caching System for Training Results
# ============================================================================
import pickle
import json
from pathlib import Path
from datetime import datetime
import threading

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MLflowTrainingCacheManager:
    """Cache manager for training and evaluation results with MLflow integration"""

    def __init__(self, base_path="training_cache"):
        self.base_path = Path(base_path)
        self.cache_lock = threading.Lock()

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Could not create cache directory: {e}")

    def save_results(self, results, filename, metadata=None):
        """Save training/evaluation results to cache"""
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

                # Log to MLflow if available
                if HAS_MLFLOW:
                    try:
                        with mlflow.start_run(run_name=f"Cache_Save_{filename}", nested=True):
                            mlflow.log_param("cache_filename", filename)
                            mlflow.log_param("cache_timestamp", metadata['timestamp'])
                            mlflow.log_artifact(str(cache_file), "cache")
                            mlflow.log_artifact(str(meta_file), "cache_metadata")
                    except Exception as e:
                        print(f"⚠️ MLflow logging failed: {e}")

                print(f"💾 Cached results: {filename}")
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

                # Load results
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)

                # Load metadata
                metadata = None
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)

                print(f"📂 Loaded cached results: {filename}")
                return results, metadata
            except Exception as e:
                print(f"❌ Cache load failed: {e}")
                return None, None

    def results_exist(self, filename):
        """Check if cached results exist"""
        cache_file = self.base_path / f"{filename}.pkl"
        return cache_file.exists()

    def clear_cache(self, filename=None):
        """Clear cache files"""
        try:
            if filename:
                cache_file = self.base_path / f"{filename}.pkl"
                meta_file = self.base_path / f"{filename}_meta.json"
                cache_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
                print(f"🗑️  Cleared cache: {filename}")
            else:
                import shutil
                shutil.rmtree(self.base_path, ignore_errors=True)
                print(f"🗑️  Cleared all cache")
        except Exception as e:
            print(f"❌ Clear cache failed: {e}")
