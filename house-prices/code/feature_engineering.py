#!/usr/bin/env python3
"""
COMPREHENSIVE FEATURE ENGINEERING
=================================
Advanced feature engineering for house prices with rigorous CV methodology.
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
    """
    Comprehensive feature engineering
    
    Args:
        df: Input dataframe
        neighborhood_stats: Precomputed neighborhood statistics (for test set)
        is_train: Whether this is training data (for computing neighborhood stats)
    
    Returns:
        Engineered dataframe and neighborhood_stats (if is_train=True)
    """
    df_eng = df.copy()
    
    # 1. SPATIAL EFFICIENCY RATIOS
    df_eng['LivingAreaEfficiency'] = df_eng['GrLivArea'] / df_eng['LotArea']
    
    # Basement finish ratio (handle divide by zero)
    total_bsmt = df_eng['TotalBsmtSF'].replace(0, np.nan)
    df_eng['BasementFinishRatio'] = (df_eng['BsmtFinSF1'] + df_eng['BsmtFinSF2']) / total_bsmt
    df_eng['BasementFinishRatio'] = df_eng['BasementFinishRatio'].fillna(0)
    
    # Second floor ratio
    total_above = df_eng['1stFlrSF'] + df_eng['2ndFlrSF']
    df_eng['SecondFloorRatio'] = df_eng['2ndFlrSF'] / total_above.replace(0, np.nan)
    df_eng['SecondFloorRatio'] = df_eng['SecondFloorRatio'].fillna(0)
    
    # Garage efficiency
    garage_cars = df_eng['GarageCars'].replace(0, np.nan)
    df_eng['GarageEfficiency'] = df_eng['GarageArea'] / garage_cars
    df_eng['GarageEfficiency'] = df_eng['GarageEfficiency'].fillna(0)
    
    # Total porch area
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    df_eng['TotalPorchArea'] = df_eng[porch_cols].sum(axis=1)
    
    # 2. AGE & DEPRECIATION FEATURES
    df_eng['HouseAge'] = df_eng['YrSold'] - df_eng['YearBuilt']
    df_eng['YearsSinceRemodel'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
    df_eng['WasRemodeled'] = (df_eng['YearRemodAdd'] != df_eng['YearBuilt']).astype(int)
    
    # Garage age difference (negative means garage is newer)
    df_eng['GarageYrBlt'] = df_eng['GarageYrBlt'].fillna(df_eng['YearBuilt'])
    df_eng['GarageAgeDiff'] = df_eng['YearBuilt'] - df_eng['GarageYrBlt']
    
    # Non-linear age effects
    df_eng['AgeSquared'] = df_eng['HouseAge'] ** 2
    df_eng['NewHouse'] = (df_eng['HouseAge'] <= 5).astype(int)
    
    # 3. QUALITY COMPOSITES
    df_eng['OverallScore'] = df_eng['OverallQual'] * df_eng['OverallCond']
    
    # Encode quality features
    quality_encoding = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    
    exter_qual = encode_ordinal_feature(df_eng['ExterQual'], quality_encoding)
    exter_cond = encode_ordinal_feature(df_eng['ExterCond'], quality_encoding)
    df_eng['ExteriorScore'] = exter_qual * exter_cond
    
    # Basement quality score
    bsmt_qual = encode_ordinal_feature(df_eng['BsmtQual'], quality_encoding)
    bsmt_cond = encode_ordinal_feature(df_eng['BsmtCond'], quality_encoding)
    bsmt_exposure = encode_ordinal_feature(df_eng['BsmtExposure'], {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1})
    df_eng['BasementScore'] = bsmt_qual * bsmt_cond * bsmt_exposure
    
    # Kitchen quality
    kitchen_qual = encode_ordinal_feature(df_eng['KitchenQual'], quality_encoding)
    
    # 4. FUNCTIONAL SPACE METRICS
    df_eng['TotalFinishedSF'] = df_eng['GrLivArea'] + df_eng['BsmtFinSF1'] + df_eng['BsmtFinSF2']
    
    # Total bathrooms
    df_eng['TotalBathrooms'] = (df_eng['FullBath'] + 0.5 * df_eng['HalfBath'] + 
                               df_eng['BsmtFullBath'] + 0.5 * df_eng['BsmtHalfBath'])
    
    # Kitchen-Bath score
    df_eng['KitchenBathScore'] = kitchen_qual * df_eng['TotalBathrooms']
    
    # Bedroom to bathroom ratio
    bedrooms = df_eng['BedroomAbvGr'].replace(0, np.nan)
    df_eng['BedroomBathRatio'] = bedrooms / df_eng['TotalBathrooms'].replace(0, np.nan)
    df_eng['BedroomBathRatio'] = df_eng['BedroomBathRatio'].fillna(0)
    
    # Average room size
    total_rooms = df_eng['TotRmsAbvGrd'].replace(0, np.nan)
    df_eng['RoomSize'] = df_eng['GrLivArea'] / total_rooms
    df_eng['RoomSize'] = df_eng['RoomSize'].fillna(0)
    
    # Unfinished potential
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
    
    # Quality-Premium interaction
    df_eng['QualityPremiumInteraction'] = df_eng['OverallQual'] * df_eng['PremiumCount']
    
    # 6. MARKET TIMING
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df_eng['SeasonSold'] = df_eng['MoSold'].map(season_map)
    
    # Quick sale indicator
    if 'SaleCondition' in df_eng.columns:
        df_eng['QuickSale'] = (df_eng['SaleCondition'] != 'Normal').astype(int)
    
    # 7. NEIGHBORHOOD STATISTICS (computed on training data)
    if is_train and 'SalePrice' in df_eng.columns:
        # Compute neighborhood statistics from training data
        neighborhood_stats = {}
        
        # Median price by neighborhood
        neighborhood_stats['median_price'] = df_eng.groupby('Neighborhood')['SalePrice'].median()
        
        # Average quality by neighborhood
        neighborhood_stats['avg_qual'] = df_eng.groupby('Neighborhood')['OverallQual'].mean()
        
        # Average lot area by neighborhood
        neighborhood_stats['avg_lotarea'] = df_eng.groupby('Neighborhood')['LotArea'].mean()
        
        # Price per sqft by neighborhood
        df_eng['PricePerSqFt'] = df_eng['SalePrice'] / df_eng['GrLivArea']
        neighborhood_stats['avg_price_per_sqft'] = df_eng.groupby('Neighborhood')['PricePerSqFt'].median()
        
    # Apply neighborhood statistics
    if neighborhood_stats is not None:
        df_eng['NeighborhoodMedianPrice'] = df_eng['Neighborhood'].map(neighborhood_stats['median_price'])
        df_eng['NeighborhoodAvgQual'] = df_eng['Neighborhood'].map(neighborhood_stats['avg_qual'])
        df_eng['NeighborhoodAvgLotArea'] = df_eng['Neighborhood'].map(neighborhood_stats['avg_lotarea'])
        df_eng['NeighborhoodAvgPricePerSqFt'] = df_eng['Neighborhood'].map(neighborhood_stats['avg_price_per_sqft'])
        
        # House vs neighborhood comparisons
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

def prepare_features_for_catboost(train, test, neighborhood_stats=None):
    """Prepare features with engineering for CatBoost"""
    
    # Apply feature engineering
    if neighborhood_stats is None:
        train_eng, neighborhood_stats = engineer_features(train, is_train=True)
    else:
        train_eng, _ = engineer_features(train, neighborhood_stats, is_train=True)
    
    test_eng = engineer_features(test, neighborhood_stats, is_train=False)
    
    # Combine and handle missing values
    all_data = pd.concat([train_eng, test_eng], ignore_index=True, sort=False)
    
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna('Unknown', inplace=True)
    
    train_processed = all_data.iloc[:len(train_eng)].reset_index(drop=True)
    test_processed = all_data.iloc[len(train_eng):].reset_index(drop=True)
    
    feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice', 'PricePerSqFt']]
    X_train = train_processed[feature_cols]
    X_test = test_processed[feature_cols]
    
    categorical_feature_indices = []
    for i, col in enumerate(X_train.columns):
        if X_train[col].dtype == 'object':
            categorical_feature_indices.append(i)
    
    return X_train, X_test, categorical_feature_indices, neighborhood_stats

def box_cox_transform(y, lam):
    """Apply Box-Cox transformation with given lambda"""
    y_positive = y + 1  # Ensure positivity
    if abs(lam) < 1e-6:  # λ ≈ 0, use log
        return np.log(y_positive)
    else:
        return (np.power(y_positive, lam) - 1) / lam

def inverse_box_cox_transform(y_transformed, lam):
    """Inverse Box-Cox transformation"""
    if abs(lam) < 1e-6:  # λ ≈ 0, use exp
        return np.exp(y_transformed) - 1
    else:
        return np.power(lam * y_transformed + 1, 1/lam) - 1

def find_optimal_box_cox_lambda(y):
    """Find optimal Box-Cox lambda parameter"""
    y_positive = y + 1  # Ensure positivity
    transformed_data, fitted_lambda = stats.boxcox(y_positive)
    return fitted_lambda

def rigorous_cv_with_feature_engineering(train, test):
    """
    Rigorous CV with feature engineering applied within each fold
    """
    print("=" * 60)
    print("FEATURE ENGINEERING + BOX-COX CV")
    print("=" * 60)
    
    params = {
        'objective': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'iterations': 1000,
        'early_stopping_rounds': 100,
        'random_seed': 42,
        'verbose': False,
        'use_best_model': True
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_transformed = []
    cv_scores_original = []
    lambdas_used = []
    feature_counts = []
    
    print("Cross-validation with feature engineering within each fold:")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train), 1):
        # Split original data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Apply feature engineering within fold
        X_train_fold, X_test_fold, cat_features, neighborhood_stats = prepare_features_for_catboost(train_fold, test)
        X_val_fold = engineer_features(val_fold, neighborhood_stats, is_train=False)
        
        # Prepare validation features
        feature_cols = [col for col in X_train_fold.columns]
        X_val_fold = X_val_fold[feature_cols]
        
        # Handle any missing values in validation
        for col in X_val_fold.columns:
            if X_val_fold[col].dtype in ['int64', 'float64']:
                X_val_fold[col] = X_val_fold[col].fillna(X_val_fold[col].median())
            else:
                X_val_fold[col] = X_val_fold[col].fillna('Unknown')
        
        # Target variables
        y_train_fold = train_fold['SalePrice']
        y_val_original = val_fold['SalePrice']
        
        # Find optimal Box-Cox lambda
        optimal_lambda = find_optimal_box_cox_lambda(y_train_fold)
        lambdas_used.append(optimal_lambda)
        
        # Apply Box-Cox transformation
        y_train_transformed = box_cox_transform(y_train_fold, optimal_lambda)
        y_val_transformed = box_cox_transform(y_val_original, optimal_lambda)
        
        # Train model
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_train_fold, y_train_transformed, 
                 eval_set=(X_val_fold, y_val_transformed), verbose=False)
        
        # Predict
        pred_transformed = model.predict(X_val_fold)
        rmse_transformed = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores_transformed.append(rmse_transformed)
        
        # Transform back to original space
        pred_original = inverse_box_cox_transform(pred_transformed, optimal_lambda)
        pred_original = np.maximum(pred_original, 1000)  # Ensure positive prices
        
        rmse_original = np.sqrt(mean_squared_error(y_val_original, pred_original))
        cv_scores_original.append(rmse_original)
        
        feature_counts.append(len(X_train_fold.columns))
        
        print(f"  Fold {fold}: {len(X_train_fold.columns)} features, λ={optimal_lambda:.4f}, "
              f"RMSE_trans={rmse_transformed:.4f}, RMSE_orig=${rmse_original:,.0f}")
    
    cv_mean_transformed = np.mean(cv_scores_transformed)
    cv_std_transformed = np.std(cv_scores_transformed)
    cv_mean_original = np.mean(cv_scores_original)
    cv_std_original = np.std(cv_scores_original)
    mean_lambda = np.mean(lambdas_used)
    avg_features = np.mean(feature_counts)
    
    print(f"\nFeature Engineering Results:")
    print(f"  Average features: {avg_features:.0f}")
    print(f"  Average λ: {mean_lambda:.4f}")
    print(f"  Transformed space: {cv_mean_transformed:.4f} ± {cv_std_transformed:.4f}")
    print(f"  Original space: ${cv_mean_original:,.0f} ± ${cv_std_original:,.0f}")
    
    return cv_mean_transformed, cv_std_transformed, cv_mean_original, cv_std_original, mean_lambda

def compare_with_baseline(train, test):
    """Compare with baseline Box-Cox model (no feature engineering)"""
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON (Box-Cox only, minimal features)")
    print("=" * 60)
    
    # Basic feature preparation (no engineering)
    all_data = pd.concat([train, test], ignore_index=True, sort=False)
    
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna('Unknown', inplace=True)
    
    train_processed = all_data.iloc[:len(train)].reset_index(drop=True)
    
    feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice']]
    X_train = train_processed[feature_cols]
    
    categorical_feature_indices = []
    for i, col in enumerate(X_train.columns):
        if X_train[col].dtype == 'object':
            categorical_feature_indices.append(i)
    
    params = {
        'objective': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'iterations': 1000,
        'early_stopping_rounds': 100,
        'random_seed': 42,
        'verbose': False,
        'use_best_model': True
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_orig, y_val_orig = train.iloc[train_idx]['SalePrice'], train.iloc[val_idx]['SalePrice']
        
        # Box-Cox transformation
        lam = find_optimal_box_cox_lambda(y_tr_orig)
        y_tr = box_cox_transform(y_tr_orig, lam)
        y_val = box_cox_transform(y_val_orig, lam)
        
        model = CatBoostRegressor(cat_features=categorical_feature_indices, **params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        cv_scores.append(rmse)
    
    baseline_cv = np.mean(cv_scores)
    
    print(f"  Baseline features: {len(X_train.columns)}")
    print(f"  Baseline CV: {baseline_cv:.4f}")
    
    return baseline_cv

def main():
    print("=" * 60)
    print("COMPREHENSIVE FEATURE ENGINEERING TEST")
    print("=" * 60)
    
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Show original target distribution
    y = train['SalePrice']
    print(f"\nOriginal target statistics:")
    print(f"  Min: ${y.min():,.0f}")
    print(f"  Max: ${y.max():,.0f}")
    print(f"  Mean: ${y.mean():,.0f}")
    print(f"  Skewness: {y.skew():.3f}")
    
    # Test with feature engineering
    eng_cv_trans, eng_cv_std, eng_cv_orig, eng_cv_std_orig, eng_lambda = rigorous_cv_with_feature_engineering(train, test)
    
    # Compare with baseline
    baseline_cv = compare_with_baseline(train, test)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"Feature Engineering CV: {eng_cv_trans:.4f} ± {eng_cv_std:.4f}")
    print(f"Baseline CV:           {baseline_cv:.4f}")
    print(f"")
    improvement = (baseline_cv - eng_cv_trans) / baseline_cv * 100
    print(f"Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("✓ Feature engineering improves performance!")
    else:
        print("✗ Feature engineering does not improve performance")

if __name__ == "__main__":
    main()