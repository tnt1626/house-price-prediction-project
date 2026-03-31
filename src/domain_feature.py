import numpy

def add_domain_features(df):
    """Phiên bản tinh gọn: Tạo các đặc trưng bất động sản quan trọng"""
    df = df.copy()

    # 1. Tổng diện tích & Số phòng tắm (Dùng fillna(0) trực tiếp)
    df["TotalSF"] = df[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].fillna(0).sum(axis=1)
    df["TotalBath"] = (df["FullBath"].fillna(0) + 0.5*df["HalfBath"].fillna(0) +
                       df["BsmtFullBath"].fillna(0) + 0.5*df["BsmtHalfBath"].fillna(0))

    # 2. Tuổi thọ nhà (Tính toán theo Vector)
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # 3. Đặc trưng nhị phân & Tỷ lệ
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
    df["TotalPorchSF"] = df[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]].fillna(0).sum(axis=1)

    # 4. Tương tác quan trọng (Quality * Area)
    if "OverallQual" in df.columns:
        df["Quality_Area_Interaction"] = df["OverallQual"] * df["GrLivArea"]

    # 5. Xử lý tính chu kỳ (Tháng bán)
    if "MoSold" in df.columns:
        df["MoSold_sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
        df["MoSold_cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)

    # 6. Gom nhóm đặc trưng (Interaction)
    if "Neighborhood" in df.columns and "BldgType" in df.columns:
        df["Loc_Type"] = df["Neighborhood"].astype(str) + "_" + df["BldgType"].astype(str)

    # 7. Giới hạn ngoại lai (Clipping)
    if "LotArea" in df.columns:
        df["LotArea_clip"] = df["LotArea"].clip(upper=df["LotArea"].quantile(0.99))

    return df
