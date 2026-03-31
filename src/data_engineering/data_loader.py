import pandas as pd
import mlflow
import os

class DataLoader:
    """
    Class này chịu trách nhiệm nạp dữ liệu và thực hiện các bước dọn dẹp cơ bản
    giống như trong Notebook House_price_prediction_project_by_Danh.
    """
    def __init__(self, file_path="/data/train-house-prices-advanced-regression-techniques.csv"):
        self.file_path = file_path

    def load_data(self):
        # 1. Importing Dataset (Tương ứng cell 10)
        if not os.path.exists(self.file_path):
            # Fallback nếu bạn chạy local thay vì Colab
            self.file_path = "train-house-prices-advanced-regression-techniques.csv"
            
        print(f"📂 Đang nạp dữ liệu từ: {self.file_path}")
        house_df = pd.read_csv(self.file_path)
        
        # 2. Ghi nhật ký quá trình load dữ liệu vào MLflow (Tương ứng cell 11)
        with mlflow.start_run(run_name="Data_Loading_Info", nested=True):
            rows, cols = house_df.shape
            total_missing = house_df.isnull().sum().sum()

            # Log tham số (Params)
            mlflow.log_params({
                "dataset_shape": f"{rows}x{cols}",
                "total_columns": cols,
                "total_rows": rows
            })

            # Log chỉ số (Metrics)
            mlflow.log_metric("missing_values_total", total_missing)
            mlflow.log_metric("missing_percentage", (total_missing / (rows * cols)) * 100)

            print(f"✅ Đã log thông tin house_df ({rows} dòng, {cols} cột) lên MLflow.")

        # 3. Remove ID column (Tương ứng cell 12)
        if 'Id' in house_df.columns:
            house_df = house_df.drop('Id', axis=1)
            print(f"🧹 Đã loại bỏ cột 'Id'. Kích thước mới: {house_df.shape}")
        
        return house_df

if __name__ == "__main__":
    # Test nhanh module
    loader = DataLoader()
    df = loader.load_data()
    print(df.head())