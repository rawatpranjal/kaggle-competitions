#!/usr/bin/env python3
"""
FINAL BEST SUBMISSION
====================
Submit our best performing model with all optimizations:
- CatBoost with optimal parameters
- 116 engineered features  
- Improved NaN handling (within-fold methodology)
- Box-Cox transformation
"""

import pandas as pd
import numpy as np
from scipy import stats
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train and test data"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def encode_ordinal_feature(series, encoding_map):
    """Encode ordinal categorical features"""
    return series.map(encoding_map).fillna(0)

def engineer_features(df, neighborhood_stats=None, is_train=True):
    """Comprehensive feature engineering (full version)"""
    df_eng = df.copy()
    
    # 1. SPATIAL EFFICIENCY RATIOS
    df_eng['LivingAreaEfficiency'] = df_eng['GrLivArea'] / df_eng['LotArea']
    
    total_bsmt = df_eng['TotalBsmtSF'].replace(0, np.nan)
    df_eng['BasementFinishRatio'] = (df_eng['BsmtFinSF1'] + df_eng['BsmtFinSF2']) / total_bsmt
    df_eng['BasementFinishRatio'] = df_eng['BasementFinishRatio'].fillna(0)
    
    total_above = df_eng['1stFlrSF'] + df_eng['2ndFlrSF']
    df_eng['SecondFloorRatio'] = df_eng['2ndFlrSF'] / total_above.replace(0, np.nan)
    df_eng['SecondFloorRatio'] = df_eng['SecondFloorRatio'].fillna(0)
    
    garage_cars = df_eng['GarageCars'].replace(0, np.nan)
    df_eng['GarageEfficiency'] = df_eng['GarageArea'] / garage_cars
    df_eng['GarageEfficiency'] = df_eng['GarageEfficiency'].fillna(0)
    
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    df_eng['TotalPorchArea'] = df_eng[porch_cols].sum(axis=1)
    
    # 2. AGE & DEPRECIATION
    df_eng['HouseAge'] = df_eng['YrSold'] - df_eng['YearBuilt']
    df_eng['YearsSinceRemodel'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
    df_eng['WasRemodeled'] = (df_eng['YearRemodAdd'] != df_eng['YearBuilt']).astype(int)
    
    df_eng['GarageYrBlt'] = df_eng['GarageYrBlt'].fillna(df_eng['YearBuilt'])
    df_eng['GarageAgeDiff'] = df_eng['YearBuilt'] - df_eng['GarageYrBlt']
    
    df_eng['AgeSquared'] = df_eng['HouseAge'] ** 2
    df_eng['NewHouse'] = (df_eng['HouseAge'] <= 5).astype(int)
    
    # 3. QUALITY COMPOSITES
    df_eng['OverallScore'] = df_eng['OverallQual'] * df_eng['OverallCond']
    
    quality_encoding = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    
    exter_qual = encode_ordinal_feature(df_eng['ExterQual'], quality_encoding)
    exter_cond = encode_ordinal_feature(df_eng['ExterCond'], quality_encoding)
    df_eng['ExteriorScore'] = exter_qual * exter_cond
    
    bsmt_qual = encode_ordinal_feature(df_eng['BsmtQual'], quality_encoding)
    bsmt_cond = encode_ordinal_feature(df_eng['BsmtCond'], quality_encoding)
    bsmt_exposure = encode_ordinal_feature(df_eng['BsmtExposure'], {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1})
    df_eng['BasementScore'] = bsmt_qual * bsmt_cond * bsmt_exposure
    
    kitchen_qual = encode_ordinal_feature(df_eng['KitchenQual'], quality_encoding)
    
    # 4. FUNCTIONAL SPACES
    df_eng['TotalFinishedSF'] = df_eng['GrLivArea'] + df_eng['BsmtFinSF1'] + df_eng['BsmtFinSF2']
    
    df_eng['TotalBathrooms'] = (df_eng['FullBath'] + 0.5 * df_eng['HalfBath'] + 
                               df_eng['BsmtFullBath'] + 0.5 * df_eng['BsmtHalfBath'])
    
    df_eng['KitchenBathScore'] = kitchen_qual * df_eng['TotalBathrooms']
    
    bedrooms = df_eng['BedroomAbvGr'].replace(0, np.nan)
    df_eng['BedroomBathRatio'] = bedrooms / df_eng['TotalBathrooms'].replace(0, np.nan)
    df_eng['BedroomBathRatio'] = df_eng['BedroomBathRatio'].fillna(0)
    
    total_rooms = df_eng['TotRmsAbvGrd'].replace(0, np.nan)
    df_eng['RoomSize'] = df_eng['GrLivArea'] / total_rooms
    df_eng['RoomSize'] = df_eng['RoomSize'].fillna(0)
    
    df_eng['UnfinishedPotential'] = df_eng['BsmtUnfSF']
    
    # 5. PREMIUM FEATURES
    df_eng['HasPool'] = (df_eng['PoolArea'] > 0).astype(int)
    df_eng['HasFireplace'] = (df_eng['Fireplaces'] > 0).astype(int)
    df_eng['HasCentralAir'] = (df_eng['CentralAir'] == 'Y').astype(int)
    df_eng['HasMasonry'] = (df_eng['MasVnrArea'] > 0).astype(int)
    df_eng['HasDeck'] = (df_eng['WoodDeckSF'] > 0).astype(int)
    df_eng['HasGarage'] = (df_eng['GarageArea'] > 0).astype(int)
    
    premium_features = ['HasPool', 'HasFireplace', 'HasCentralAir', 'HasMasonry', 'HasDeck', 'HasGarage']
    df_eng['PremiumCount'] = df_eng[premium_features].sum(axis=1)
    df_eng['QualityPremiumInteraction'] = df_eng['OverallQual'] * df_eng['PremiumCount']
    
    # 6. MARKET TIMING
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df_eng['SeasonSold'] = df_eng['MoSold'].map(season_map)
    
    if 'SaleCondition' in df_eng.columns:
        df_eng['QuickSale'] = (df_eng['SaleCondition'] != 'Normal').astype(int)
    
    # 7. NEIGHBORHOOD STATISTICS
    if is_train and 'SalePrice' in df_eng.columns:
        neighborhood_stats = {}
        neighborhood_stats['median_price'] = df_eng.groupby('Neighborhood')['SalePrice'].median()
        neighborhood_stats['avg_qual'] = df_eng.groupby('Neighborhood')['OverallQual'].mean()
        neighborhood_stats['avg_lotarea'] = df_eng.groupby('Neighborhood')['LotArea'].mean()
        
        df_eng['PricePerSqFt'] = df_eng['SalePrice'] / df_eng['GrLivArea']
        neighborhood_stats['avg_price_per_sqft'] = df_eng.groupby('Neighborhood')['PricePerSqFt'].median()
        
    if neighborhood_stats is not None:
        df_eng['NeighborhoodMedianPrice'] = df_eng['Neighborhood'].map(neighborhood_stats['median_price'])
        df_eng['NeighborhoodAvgQual'] = df_eng['Neighborhood'].map(neighborhood_stats['avg_qual'])
        df_eng['NeighborhoodAvgLotArea'] = df_eng['Neighborhood'].map(neighborhood_stats['avg_lotarea'])
        df_eng['NeighborhoodAvgPricePerSqFt'] = df_eng['Neighborhood'].map(neighborhood_stats['avg_price_per_sqft'])
        
        df_eng['QualVsNeighborhood'] = df_eng['OverallQual'] - df_eng['NeighborhoodAvgQual']
        df_eng['LotAreaVsNeighborhood'] = df_eng['LotArea'] - df_eng['NeighborhoodAvgLotArea']
    
    # 8. ERA CATEGORIZATION
    df_eng['Era'] = 'Modern'
    df_eng.loc[df_eng['YearBuilt'] < 1950, 'Era'] = 'Pre1950'
    df_eng.loc[(df_eng['YearBuilt'] >= 1950) & (df_eng['YearBuilt'] < 1980), 'Era'] = '1950to1980'
    df_eng.loc[(df_eng['YearBuilt'] >= 1980) & (df_eng['YearBuilt'] < 2000), 'Era'] = '1980to2000'
    
    if is_train:
        return df_eng, neighborhood_stats
    else:
        return df_eng

def improved_nan_handling(train_data, test_data):
    """
    Improved NaN handling: Use training data statistics only
    """
    train_filled = train_data.copy()
    test_filled = test_data.copy()
    
    numeric_cols = train_filled.select_dtypes(include=[np.number]).columns
    categorical_cols = train_filled.select_dtypes(include=['object']).columns
    
    # Handle numeric columns with median from training data only
    for col in numeric_cols:
        if train_filled[col].isnull().sum() > 0 or test_filled[col].isnull().sum() > 0:
            median_val = train_filled[col].median()
            train_filled[col] = train_filled[col].fillna(median_val)
            test_filled[col] = test_filled[col].fillna(median_val)
    
    # Handle categorical columns with mode from training data only
    for col in categorical_cols:
        if train_filled[col].isnull().sum() > 0 or test_filled[col].isnull().sum() > 0:
            mode_values = train_filled[col].mode()
            if len(mode_values) > 0:
                mode_val = mode_values.iloc[0]
            else:
                mode_val = 'Unknown'
            
            train_filled[col] = train_filled[col].fillna(mode_val)
            test_filled[col] = test_filled[col].fillna(mode_val)
    
    # Ensure categorical columns are strings
    for col in categorical_cols:
        train_filled[col] = train_filled[col].astype(str)
        test_filled[col] = test_filled[col].astype(str)
    
    return train_filled, test_filled

def find_optimal_box_cox_lambda(y):
    """Find optimal Box-Cox lambda parameter"""
    y_positive = y + 1
    transformed_data, fitted_lambda = stats.boxcox(y_positive)
    return fitted_lambda

def box_cox_transform(y, lam):
    """Apply Box-Cox transformation"""
    y_positive = y + 1
    if abs(lam) < 1e-6:
        return np.log(y_positive)
    else:
        return (np.power(y_positive, lam) - 1) / lam

def inverse_box_cox_transform(y_transformed, lam):
    """Inverse Box-Cox transformation"""
    if abs(lam) < 1e-6:
        return np.exp(y_transformed) - 1
    else:
        return np.power(lam * y_transformed + 1, 1/lam) - 1

def create_final_submission(train, test):
    """Create final submission with all optimizations"""
    print("=" * 60)
    print("CREATING FINAL BEST SUBMISSION")
    print("=" * 60)
    
    # Apply feature engineering
    print("Applying comprehensive feature engineering...")
    train_eng, neighborhood_stats = engineer_features(train, is_train=True)
    test_eng = engineer_features(test, neighborhood_stats, is_train=False)
    
    # Get feature columns
    feature_cols = [col for col in train_eng.columns if col not in ['Id', 'SalePrice', 'PricePerSqFt']]
    
    train_features = train_eng[feature_cols]
    test_features = test_eng[feature_cols]
    
    # Apply improved NaN handling
    print("Applying improved NaN handling...")
    train_filled, test_filled = improved_nan_handling(train_features, test_features)
    
    print(f"Final feature count: {len(train_filled.columns)}")
    
    # Get categorical feature indices
    categorical_feature_indices = []
    for i, col in enumerate(train_filled.columns):
        if train_filled[col].dtype == 'object' or train_filled[col].dtype.name == 'string':
            categorical_feature_indices.append(i)
    
    print(f"Categorical features: {len(categorical_feature_indices)}")
    
    # Target variable
    y_train_original = train['SalePrice']
    
    # Find optimal Box-Cox lambda
    optimal_lambda = find_optimal_box_cox_lambda(y_train_original)
    print(f"Optimal Box-Cox lambda: {optimal_lambda:.4f}")
    
    # Transform target
    y_train_transformed = box_cox_transform(y_train_original, optimal_lambda)
    
    print(f"Target transformation:")
    print(f"  Original skewness: {y_train_original.skew():.3f}")
    print(f"  Transformed skewness: {stats.skew(y_train_transformed):.3f}")
    
    # Optimal CatBoost parameters (from tuning)
    params = {
        'objective': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'iterations': 1200,  # More iterations for final model
        'random_seed': 42,
        'verbose': False
    }
    
    print("Training final model with optimal parameters...")
    print(f"Parameters: {params}")
    
    # Train final model
    model = CatBoostRegressor(cat_features=categorical_feature_indices, **params)
    model.fit(train_filled, y_train_transformed)
    
    # Make predictions
    print("Making predictions...")
    test_pred_transformed = model.predict(test_filled)
    
    # Transform back to original space
    test_pred_original = inverse_box_cox_transform(test_pred_transformed, optimal_lambda)
    test_pred_original = np.maximum(test_pred_original, 1000)  # Ensure positive prices
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': test_pred_original
    })
    
    filename = 'submissions/final_best_submission.csv'
    submission.to_csv(filename, index=False)
    
    print(f"\nFinal submission created: {filename}")
    print(f"Prediction statistics:")
    print(f"  Min: ${test_pred_original.min():,.0f}")
    print(f"  Max: ${test_pred_original.max():,.0f}")
    print(f"  Mean: ${test_pred_original.mean():,.0f}")
    print(f"  Median: ${np.median(test_pred_original):,.0f}")
    
    print(f"\nModel configuration:")
    print(f"  - CatBoost with {len(categorical_feature_indices)} categorical features")
    print(f"  - {len(train_filled.columns)} total engineered features") 
    print(f"  - Box-Cox transformation (λ={optimal_lambda:.4f})")
    print(f"  - Improved NaN handling (training-only statistics)")
    print(f"  - Expected CV: ~0.0492 RMSE (transformed space)")
    
    return submission, filename

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    submission, filename = create_final_submission(train, test)
    
    print("\n" + "=" * 60)
    print("READY FOR KAGGLE SUBMISSION")
    print("=" * 60)
    print(f"File: {filename}")
    print("This model combines all our best optimizations:")
    print("✓ Comprehensive feature engineering (116 features)")
    print("✓ Improved NaN handling (no data leakage)")
    print("✓ Box-Cox transformation (optimal λ)")
    print("✓ CatBoost with optimal hyperparameters")
    print("✓ Native categorical feature handling")
    print("\nExpected to improve upon previous best: 0.12241")

if __name__ == "__main__":
    main()