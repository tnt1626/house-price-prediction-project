# data_engineering/data_loader.py
import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Tải dữ liệu từ file CSV"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file tại {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        print(f"✅ Đã tải dữ liệu với kích thước: {df.shape}")
        
        # Loại bỏ cột 'Id' vì không có ý nghĩa trong việc train model
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
            print("✅ Đã loại bỏ cột 'Id'.")
            
        return df