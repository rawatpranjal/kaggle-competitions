#!/usr/bin/env python3
"""
FIXED NaN HANDLING TEST
=======================
Test improved NaN handling with proper categorical feature handling for CatBoost.
"""

import pandas as pd
import numpy as np
from scipy import stats
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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
    """Comprehensive feature engineering"""
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

def improved_nan_handling_within_fold(train_data, val_data, test_data):
    """
    Improved NaN handling: Calculate imputation values from training fold only
    Then apply to train/val/test consistently
    """
    # Create copies
    train_filled = train_data.copy()
    val_filled = val_data.copy() 
    test_filled = test_data.copy()
    
    # Identify columns by type
    numeric_cols = train_filled.select_dtypes(include=[np.number]).columns
    categorical_cols = train_filled.select_dtypes(include=['object']).columns
    
    imputation_values = {}
    
    # Handle numeric columns with median from training data only
    for col in numeric_cols:
        if train_filled[col].isnull().sum() > 0 or val_filled[col].isnull().sum() > 0 or test_filled[col].isnull().sum() > 0:
            # Calculate median from training data only
            median_val = train_filled[col].median()
            imputation_values[f'numeric_{col}'] = median_val
            
            # Fill NaNs in all datasets with training median
            train_filled[col] = train_filled[col].fillna(median_val)
            val_filled[col] = val_filled[col].fillna(median_val)
            test_filled[col] = test_filled[col].fillna(median_val)
    
    # Handle categorical columns with mode from training data only
    for col in categorical_cols:
        if train_filled[col].isnull().sum() > 0 or val_filled[col].isnull().sum() > 0 or test_filled[col].isnull().sum() > 0:
            # Calculate mode from training data only
            mode_values = train_filled[col].mode()
            if len(mode_values) > 0:
                mode_val = mode_values.iloc[0]  # Take first mode if multiple
            else:
                mode_val = 'Unknown'
            
            imputation_values[f'categorical_{col}'] = mode_val
            
            # Fill NaNs in all datasets with training mode
            train_filled[col] = train_filled[col].fillna(mode_val)
            val_filled[col] = val_filled[col].fillna(mode_val)  
            test_filled[col] = test_filled[col].fillna(mode_val)
    
    # Ensure no NaNs remain and convert categorical to string
    for col in categorical_cols:
        train_filled[col] = train_filled[col].astype(str)
        val_filled[col] = val_filled[col].astype(str)
        test_filled[col] = test_filled[col].astype(str)
    
    return train_filled, val_filled, test_filled, imputation_values

def current_nan_handling(train_data, val_data, test_data):
    """
    Current NaN handling: Combine all data first, then impute
    (This can cause data leakage)
    """
    # Combine all data
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True, sort=False)
    
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    # Fill numeric with median of all data
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna(all_data[col].median())
    
    # Fill categorical with 'Unknown'
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna('Unknown')
        # Convert to string to avoid CatBoost issues
        all_data[col] = all_data[col].astype(str)
    
    # Split back
    train_filled = all_data.iloc[:len(train_data)].reset_index(drop=True)
    val_filled = all_data.iloc[len(train_data):len(train_data)+len(val_data)].reset_index(drop=True)
    test_filled = all_data.iloc[len(train_data)+len(val_data):].reset_index(drop=True)
    
    return train_filled, val_filled, test_filled

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

def test_improved_nan_handling(train, test):
    """Test improved NaN handling within each fold"""
    print("=" * 60)
    print("IMPROVED NaN HANDLING (within fold)")
    print("=" * 60)
    
    params = {
        'objective': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'iterations': 1000,
        'early_stopping_rounds': 100,
        'random_state': 42,
        'verbose': False,
        'use_best_model': True
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_transformed = []
    
    print("Cross-validation with improved NaN handling:")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train), 1):
        # Split original data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Apply feature engineering
        train_fold_eng, neighborhood_stats = engineer_features(train_fold, is_train=True)
        val_fold_eng = engineer_features(val_fold, neighborhood_stats, is_train=False)
        test_eng = engineer_features(test, neighborhood_stats, is_train=False)
        
        # Get feature columns
        feature_cols = [col for col in train_fold_eng.columns if col not in ['Id', 'SalePrice', 'PricePerSqFt']]
        
        train_features = train_fold_eng[feature_cols]
        val_features = val_fold_eng[feature_cols]
        test_features = test_eng[feature_cols]
        
        # Apply improved NaN handling
        train_filled, val_filled, test_filled, imputation_values = improved_nan_handling_within_fold(
            train_features, val_features, test_features
        )
        
        # Get categorical feature indices
        categorical_feature_indices = []
        for i, col in enumerate(train_filled.columns):
            if train_filled[col].dtype == 'object' or train_filled[col].dtype.name == 'string':
                categorical_feature_indices.append(i)
        
        # Target variables
        y_train_fold = train_fold['SalePrice']
        y_val_original = val_fold['SalePrice']
        
        # Box-Cox transformation
        optimal_lambda = find_optimal_box_cox_lambda(y_train_fold)
        y_train_transformed = box_cox_transform(y_train_fold, optimal_lambda)
        y_val_transformed = box_cox_transform(y_val_original, optimal_lambda)
        
        # Train model
        model = CatBoostRegressor(cat_features=categorical_feature_indices, **params)
        model.fit(train_filled, y_train_transformed, 
                 eval_set=(val_filled, y_val_transformed), verbose=False)
        
        # Predict
        pred_transformed = model.predict(val_filled)
        rmse_transformed = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores_transformed.append(rmse_transformed)
        
        n_imputed = len([k for k in imputation_values.keys()])
        print(f"  Fold {fold}: {n_imputed} features imputed, λ={optimal_lambda:.4f}, RMSE={rmse_transformed:.4f}")
    
    cv_mean = np.mean(cv_scores_transformed)
    cv_std = np.std(cv_scores_transformed)
    
    print(f"\nImproved NaN handling results:")
    print(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return cv_mean, cv_std

def test_current_nan_handling(train, test):
    """Test current NaN handling method"""
    print("\n" + "=" * 60)
    print("CURRENT NaN HANDLING (combined data)")
    print("=" * 60)
    
    params = {
        'objective': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'iterations': 1000,
        'early_stopping_rounds': 100,
        'random_state': 42,
        'verbose': False,
        'use_best_model': True
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_transformed = []
    
    print("Cross-validation with current NaN handling:")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train), 1):
        # Split original data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Apply feature engineering
        train_fold_eng, neighborhood_stats = engineer_features(train_fold, is_train=True)
        val_fold_eng = engineer_features(val_fold, neighborhood_stats, is_train=False)
        test_eng = engineer_features(test, neighborhood_stats, is_train=False)
        
        # Get feature columns
        feature_cols = [col for col in train_fold_eng.columns if col not in ['Id', 'SalePrice', 'PricePerSqFt']]
        
        train_features = train_fold_eng[feature_cols]
        val_features = val_fold_eng[feature_cols]
        test_features = test_eng[feature_cols]
        
        # Apply current NaN handling (combines all data first)
        train_filled, val_filled, test_filled = current_nan_handling(
            train_features, val_features, test_features
        )
        
        # Get categorical feature indices
        categorical_feature_indices = []
        for i, col in enumerate(train_filled.columns):
            if train_filled[col].dtype == 'object' or train_filled[col].dtype.name == 'string':
                categorical_feature_indices.append(i)
        
        # Target variables
        y_train_fold = train_fold['SalePrice']
        y_val_original = val_fold['SalePrice']
        
        # Box-Cox transformation
        optimal_lambda = find_optimal_box_cox_lambda(y_train_fold)
        y_train_transformed = box_cox_transform(y_train_fold, optimal_lambda)
        y_val_transformed = box_cox_transform(y_val_original, optimal_lambda)
        
        # Train model
        model = CatBoostRegressor(cat_features=categorical_feature_indices, **params)
        model.fit(train_filled, y_train_transformed, 
                 eval_set=(val_filled, y_val_transformed), verbose=False)
        
        # Predict
        pred_transformed = model.predict(val_filled)
        rmse_transformed = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores_transformed.append(rmse_transformed)
        
        print(f"  Fold {fold}: λ={optimal_lambda:.4f}, RMSE={rmse_transformed:.4f}")
    
    cv_mean = np.mean(cv_scores_transformed)
    cv_std = np.std(cv_scores_transformed)
    
    print(f"\nCurrent NaN handling results:")
    print(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return cv_mean, cv_std

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Analyze NaN patterns
    print(f"\nNaN analysis:")
    train_nans = train.isnull().sum().sum()
    test_nans = test.isnull().sum().sum()
    print(f"  Total NaNs in train: {train_nans}")
    print(f"  Total NaNs in test: {test_nans}")
    
    # Test improved NaN handling
    improved_cv, improved_std = test_improved_nan_handling(train, test)
    
    # Test current NaN handling
    current_cv, current_std = test_current_nan_handling(train, test)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("NaN HANDLING COMPARISON")
    print("=" * 60)
    print(f"Improved method (within fold): {improved_cv:.4f} ± {improved_std:.4f}")
    print(f"Current method (combined):     {current_cv:.4f} ± {current_std:.4f}")
    print(f"")
    
    improvement = (current_cv - improved_cv) / current_cv * 100
    print(f"Improvement: {improvement:+.2f}%")
    
    if improved_cv < current_cv:
        print("✓ Improved NaN handling helps!")
    else:
        print("✗ Improved NaN handling does not help")
    
    print(f"\nNote: Improved method prevents data leakage by using only")
    print(f"training fold statistics for imputation.")

if __name__ == "__main__":
    main()