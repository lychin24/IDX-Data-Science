import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

data_path = "insert file path to cleaned_df.csv"
geo_data_path = "insert file path to CA district geo data"

target = 'ClosePrice'

cat_col = ['Flooring', 'AssociationFeeFrequency',
        'MLSAreaMajor', 'ElementarySchool', 'SubdivisionName', 'City',
        'PurchaseContractDate', 'MiddleOrJuniorSchool', 'HighSchool',
        'HighSchoolDistrict', 'Levels', 'ListingKey', 'CloseDate',
        'PropertyType', 'ListingKeyNumeric', 'CountyOrParish',
        'PropertySubType', 'ListingId', 'ContractStatusChangeDate',
        'ListingContractDate', 'StateOrProvince', 'UnifiedDistrict']

bool_col = ['ViewYN', 'PoolPrivateYN', 'AttachedGarageYN', 'FireplaceYN', 'NewConstructionYN']

num_col = ['LivingArea', 'LotSizeSquareFeet', 'BathroomsTotalInteger',
            'Stories', 'MainLevelBedrooms', 'BedroomsTotal', 'DaysOnMarket',
            'LotLivingRatio', 'BathroomBedroomRatio', 'TotalAmenityCount', 'FlooringCount']

required_cols = ['ParkingTotal', 'Longitude', 'Latitude']


# ----------------------------
    # Load dataset
# ----------------------------
def load_dataset(path=data_path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df[df['StateOrProvince'] == 'CA']
    print('--------Dataset Loaded--------')
    return df


# ----------------------------
    # Add districts, only include properties with Unified districts
# ----------------------------
def add_districts(df, path=geo_data_path):
    districts = gpd.read_file(path)
    districts = districts.to_crs("EPSG:4326")
    district_info = districts[['DistrictName', 'DistrictType', 'geometry']].copy()

    geo_df = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs='EPSG:4326')
    df_districts = gpd.sjoin(geo_df, district_info, how="left", predicate="within").drop(columns=['index_right', 'geometry'])
    district_features = (df_districts.reset_index().pivot_table(index='index', columns='DistrictType', values='DistrictName', aggfunc='first'))
    district_split = district_features.rename(columns={'Elementary': 'ElementaryDistrict', 'High': 'HighDistrict', 'Unified': 'UnifiedDistrict'})
    df = df_districts.join(district_split).drop(columns=['DistrictType', 'DistrictName', 'ElementaryDistrict', 'HighDistrict']).drop_duplicates()
    print('--------Districts Added--------')
    return df


# ----------------------------
    # Feature engineering
# ----------------------------
def add_features(df):
    model_df = df.copy()
    model_df["LotLivingRatio"] = np.where(df["LotSizeSquareFeet"] > 0, df["LivingArea"] / df["LotSizeSquareFeet"], np.nan)
    model_df['BathroomBedroomRatio'] = np.where(df['BedroomsTotal'] > 0, df['BathroomsTotalInteger']/df['BedroomsTotal'], np.nan)

    amenity_cols = ['ViewYN', 'PoolPrivateYN', 'AttachedGarageYN', 'FireplaceYN']
    amenities = df[amenity_cols].fillna(False).astype(int)
    model_df['TotalAmenityCount'] = amenities.sum(axis=1)

    model_df['FlooringCount'] = df['Flooring'].str.split(',').str.len()
    print('--------Features Added--------')
    return model_df


# ----------------------------
    # Only include columns that are relevant to modeling
# ----------------------------
def clean_cols(df):
    keep_num = [col for col in num_col if col in df.columns]
    keep_cat = [col for col in cat_col if col in df.columns]
    keep_bool = [col for col in bool_col if col in df.columns]

    keep_cols = [target] + keep_cat + keep_bool + required_cols + keep_num
    df = df[keep_cols].copy()
    print('--------Dataset Cleaned--------')
    return df


# ----------------------------
    # Fill missing values and encode categorical variables
# ----------------------------
def preproc_df(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # ----------------------------
    # Drop rows with missing values in key columns
    # ----------------------------
    required_cols = ['ParkingTotal', 'Longitude', 'Latitude']
    train_df = train_df.dropna(subset=required_cols)
    test_df = test_df.dropna(subset=required_cols)

    # Remove invalid values
    train_df = train_df[
        (train_df["ClosePrice"] > 0) &
        (train_df["LivingArea"] > 0) &
        (train_df["BathroomsTotalInteger"] > 0) &
        (train_df["DaysOnMarket"] > 0)
    ]

    test_df = test_df[
        (test_df["ClosePrice"] > 0) &
        (test_df["LivingArea"] > 0) &
        (test_df["BathroomsTotalInteger"] > 0) &
        (test_df["DaysOnMarket"] > 0)
    ]

    # ----------------------------
    # Missing value handling
    # ----------------------------
    for col in cat_col:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("Unknown")
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("Unknown")

    for col in train_df.columns:
        if train_df[col].isna().sum() > 0:
            train_df[f"{col}_was_missing"] = train_df[col].isna().astype(int)
            test_df[f"{col}_was_missing"] = test_df[col].isna().astype(int)

    if "YearBuilt" in train_df.columns:
        median = train_df["YearBuilt"].median()
        train_df["YearBuilt"] = train_df["YearBuilt"].fillna(median)
        test_df["YearBuilt"] = test_df["YearBuilt"].fillna(median)

    if "StreetNumberNumeric" in train_df.columns:
        train_df["StreetNumberNumeric"] = train_df["StreetNumberNumeric"].fillna(-1)
        test_df["StreetNumberNumeric"] = test_df["StreetNumberNumeric"].fillna(-1)

    train_df[bool_col] = train_df[bool_col].fillna(False)
    test_df[bool_col] = test_df[bool_col].fillna(False)

    for col in num_col:
        if col in train_df.columns:
            median = train_df[col].median()
            train_df[col] = train_df[col].fillna(median)
            test_df[col] = test_df[col].fillna(median)

    if "AssociationFee" in train_df.columns:
        train_df["AssociationFee"] = train_df["AssociationFee"].fillna(0)
        test_df["AssociationFee"] = test_df["AssociationFee"].fillna(0)

    for col in ["GarageSpaces", "ParkingTotal"]:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)

    # ----------------------------
    # Boolean encoding
    # ----------------------------
    train_df[bool_col] = train_df[bool_col].astype(int)
    test_df[bool_col] = test_df[bool_col].astype(int)

    # ----------------------------
    # MultiLabel Encoding
    # ----------------------------
    multi_cols = ["Flooring", "Levels"]

    for col in multi_cols:
        train_df[col] = train_df[col].fillna("").str.split(",")
        test_df[col] = test_df[col].fillna("").str.split(",")

        mlb = MultiLabelBinarizer()

        train_encoded = pd.DataFrame(
            mlb.fit_transform(train_df[col]),
            columns=[f"{col}_{c}" for c in mlb.classes_],
            index=train_df.index
        )

        test_encoded = pd.DataFrame(
            mlb.transform(test_df[col]),
            columns=[f"{col}_{c}" for c in mlb.classes_],
            index=test_df.index
        )

        train_df = train_df.drop(columns=col).join(train_encoded)
        test_df = test_df.drop(columns=col).join(test_encoded)

    # ----------------------------
    # Ordinal Encoding
    # ----------------------------
    mapping = {
        "Unknown": 0,
        "Monthly": 1,
        "Quarterly": 2,
        "SemiAnnually": 3,
        "Annually": 4
    }

    train_df["AssociationFeeFrequency"] = train_df["AssociationFeeFrequency"].map(mapping)
    test_df["AssociationFeeFrequency"] = test_df["AssociationFeeFrequency"].map(mapping)

    # ----------------------------
    # One-Hot Encoding
    # ----------------------------
    one_hot = [
        "CountyOrParish",
        "StateOrProvince",
        "City",
        "PropertyType",
        "PropertySubType",
        "UnifiedDistrict"
    ]

    for col in one_hot:
        top = train_df[col].value_counts().head(200).index

        train_df[col] = train_df[col].where(train_df[col].isin(top), "Other")
        test_df[col] = test_df[col].where(test_df[col].isin(top), "Other")

    train_df = pd.get_dummies(train_df, columns=one_hot, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=one_hot, drop_first=True)

    # Ensure same columns
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    # ----------------------------
    # Standardization
    # ----------------------------
    scale = [
        "LivingArea",
        "LotSizeSquareFeet",
        "AssociationFee",
        "DaysOnMarket"
    ]

    scale = [c for c in scale if c in train_df.columns]

    scaler = StandardScaler()
    train_df[scale] = scaler.fit_transform(train_df[scale])
    test_df[scale] = scaler.transform(test_df[scale])

    train_df = train_df.drop(columns=cat_col, errors='ignore')
    test_df = test_df.drop(columns=cat_col, errors='ignore')

    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()

    print('--------Dataset Preprocessed--------')
    return train_df, test_df


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mdape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100


# ----------------------------
    # Split into train / test by time window
# ----------------------------
def split_train_test(model_df):
    model_df = model_df.copy()
    model_df['CloseDate'] = pd.to_datetime(model_df['CloseDate'])
    model_df = model_df.sort_values('CloseDate').reset_index(drop=True)

    def get_time_split(data, X_months):
        latest_date = data['CloseDate'].max()
        test_start_date = latest_date - pd.DateOffset(months=1)
        train_start_date = test_start_date - pd.DateOffset(months=X_months)
        test_set = data[data['CloseDate'] >= test_start_date]
        train_set = data[(data['CloseDate'] >= train_start_date) & (data['CloseDate'] < test_start_date)]
        return train_set, test_set

    train, test = get_time_split(model_df, X_months=13)
    train_df, test_df = preproc_df(train, test)

    X_train = train_df.drop(columns=['ClosePrice', 'DaysOnMarket'])
    y_train = train_df[target]
    X_test = test_df.drop(columns=['ClosePrice', 'DaysOnMarket'])
    y_test = test_df[target]
    print('--------Dataset Split--------')
    return X_train, y_train, X_test, y_test


# ----------------------------
    # Train models
# ----------------------------
def run_models(X_train, y_train, X_test, y_test):
    model_results = []

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "XGBoost": xgb.XGBRegressor(max_depth=8, learning_rate=0.10, n_estimators=700,
                                    random_state=42, n_jobs=-1)
    }

    for model_name, model in models.items():
        print(f'Fitting {model_name}')
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)

        mae = mean_absolute_error(y_test, test_preds)
        rmse = root_mean_squared_error(y_test, test_preds)

        model_results.append({
            "model": model_name,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "mae": mae,
            "rmse": rmse,
            "mape": mape(y_test, test_preds),
            "mdape": mdape(y_test, test_preds)
        })

    comparison = pd.DataFrame(model_results)
    return comparison


def main():
    df = load_dataset()
    df = add_districts(df)
    df = add_features(df)
    model_df = clean_cols(df)
    X_train, y_train, X_test, y_test = split_train_test(model_df)
    output = run_models(X_train, y_train, X_test, y_test)
    print(output)
    return output


main()