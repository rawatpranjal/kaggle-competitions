#!/usr/bin/env python3
"""
MULTIVARIATE OUTLIER DETECTION + CATBOOST
==========================================
Detect and remove multivariate outliers using multiple methods, then train CatBoost.
"""

import pandas as pd
import numpy as np
from scipy import stats
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def prepare_features_for_catboost(train, test, neighborhood_stats=None):
    """Prepare features for CatBoost"""
    
    if neighborhood_stats is None:
        train_eng, neighborhood_stats = engineer_features(train, is_train=True)
    else:
        train_eng, _ = engineer_features(train, neighborhood_stats, is_train=True)
    
    test_eng = engineer_features(test, neighborhood_stats, is_train=False)
    
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

def detect_multivariate_outliers(X_train, method='isolation', contamination=0.05):
    """
    Detect multivariate outliers using various methods
    
    Args:
        X_train: Feature matrix
        method: 'isolation', 'elliptic', 'mahalanobis', 'ensemble'
        contamination: Expected proportion of outliers
    
    Returns:
        outlier_mask: Boolean mask where True indicates outlier
        outlier_scores: Outlier scores for analysis
    """
    # Prepare numeric features for outlier detection
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_numeric = X_train[numeric_cols].copy()
    
    # Handle any remaining NaNs
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    # Standardize features for distance-based methods
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    if method == 'isolation':
        # Isolation Forest - good for high-dimensional data
        detector = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = detector.fit_predict(X_scaled)
        outlier_scores = detector.decision_function(X_scaled)
        outlier_mask = outlier_pred == -1
        
    elif method == 'elliptic':
        # Elliptic Envelope - assumes Gaussian distribution
        detector = EllipticEnvelope(contamination=contamination, random_state=42)
        outlier_pred = detector.fit_predict(X_scaled)
        outlier_scores = detector.decision_function(X_scaled)
        outlier_mask = outlier_pred == -1
        
    elif method == 'mahalanobis':
        # Mahalanobis distance - multivariate distance from center
        mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled.T)
        
        # Calculate Mahalanobis distances
        inv_cov = np.linalg.pinv(cov)  # Use pseudo-inverse for stability
        diff = X_scaled - mean
        mahal_dist = np.array([np.sqrt(np.dot(np.dot(d, inv_cov), d)) for d in diff])
        
        # Use chi-square threshold
        threshold = np.percentile(mahal_dist, (1 - contamination) * 100)
        outlier_mask = mahal_dist > threshold
        outlier_scores = -mahal_dist  # Negative for consistency (lower = more outlier-like)
        
    elif method == 'ensemble':
        # Ensemble of multiple methods
        methods = ['isolation', 'elliptic']
        scores_list = []
        masks_list = []
        
        for sub_method in methods:
            mask, scores = detect_multivariate_outliers(X_train, method=sub_method, contamination=contamination)
            scores_list.append(scores)
            masks_list.append(mask)
        
        # Combine scores (average)
        outlier_scores = np.mean(scores_list, axis=0)
        
        # Outlier if majority of methods agree
        outlier_mask = np.sum(masks_list, axis=0) >= len(methods) // 2 + 1
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outlier_mask, outlier_scores

def analyze_outliers(X_train, y_train, outlier_mask, outlier_scores):
    """Analyze detected outliers"""
    outlier_indices = np.where(outlier_mask)[0]
    n_outliers = len(outlier_indices)
    
    print(f"Outlier Analysis:")
    print(f"  Total outliers detected: {n_outliers} ({n_outliers/len(X_train)*100:.1f}%)")
    
    if n_outliers > 0:
        print(f"  Price range of outliers: ${y_train.iloc[outlier_indices].min():,.0f} - ${y_train.iloc[outlier_indices].max():,.0f}")
        print(f"  Median price of outliers: ${y_train.iloc[outlier_indices].median():,.0f}")
        print(f"  Median price of inliers: ${y_train.iloc[~outlier_mask].median():,.0f}")
        
        # Show most extreme outliers
        extreme_indices = outlier_indices[np.argsort(outlier_scores[outlier_indices])[:5]]
        print(f"  Most extreme outlier prices: {[f'${price:,.0f}' for price in y_train.iloc[extreme_indices]]}")
    
    return outlier_indices

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

def test_outlier_removal_methods(train, test):
    """Test different outlier removal methods"""
    print("=" * 60)
    print("MULTIVARIATE OUTLIER DETECTION COMPARISON")
    print("=" * 60)
    
    methods = [
        ('isolation', 0.05, "Isolation Forest (5%)"),
        ('elliptic', 0.05, "Elliptic Envelope (5%)"),
        ('mahalanobis', 0.05, "Mahalanobis Distance (5%)"),
        ('isolation', 0.03, "Isolation Forest (3%)"),
        ('ensemble', 0.05, "Ensemble Methods (5%)")
    ]
    
    results = {}
    
    for method, contamination, description in methods:
        print(f"\n" + "=" * 50)
        print(f"Testing {description}")
        print("=" * 50)
        
        cv_scores = test_outlier_method(train, test, method, contamination)
        results[description] = cv_scores
        
        print(f"{description} CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return results

def test_outlier_method(train, test, method='isolation', contamination=0.05):
    """Test specific outlier detection method with rigorous CV"""
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
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train), 1):
        # Split original data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Apply feature engineering
        train_fold_eng, neighborhood_stats = engineer_features(train_fold, is_train=True)
        val_fold_eng = engineer_features(val_fold, neighborhood_stats, is_train=False)
        test_eng = engineer_features(test, neighborhood_stats, is_train=False)
        
        # Prepare features
        all_data = pd.concat([train_fold_eng, val_fold_eng, test_eng], ignore_index=True, sort=False)
        
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        categorical_cols = all_data.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if all_data[col].isnull().sum() > 0:
                all_data[col].fillna(all_data[col].median(), inplace=True)
        
        for col in categorical_cols:
            if all_data[col].isnull().sum() > 0:
                all_data[col].fillna('Unknown', inplace=True)
        
        train_processed = all_data.iloc[:len(train_fold_eng)].reset_index(drop=True)
        val_processed = all_data.iloc[len(train_fold_eng):len(train_fold_eng)+len(val_fold_eng)].reset_index(drop=True)
        
        feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice', 'PricePerSqFt']]
        X_train_fold = train_processed[feature_cols]
        X_val_fold = val_processed[feature_cols]
        
        categorical_feature_indices = []
        for i, col in enumerate(X_train_fold.columns):
            if X_train_fold[col].dtype == 'object':
                categorical_feature_indices.append(i)
        
        y_train_fold = train_fold['SalePrice']
        y_val_original = val_fold['SalePrice']
        
        # Detect and remove outliers from training fold only
        if method is not None:
            outlier_mask, outlier_scores = detect_multivariate_outliers(
                X_train_fold, method=method, contamination=contamination
            )
            
            if fold == 1:  # Analyze outliers for first fold
                outlier_indices = analyze_outliers(X_train_fold, y_train_fold, outlier_mask, outlier_scores)
            
            # Remove outliers from training data
            X_train_clean = X_train_fold.loc[~outlier_mask].reset_index(drop=True)
            y_train_clean = y_train_fold.loc[~outlier_mask].reset_index(drop=True)
        else:
            # No outlier removal - use all training data
            outlier_mask = np.zeros(len(X_train_fold), dtype=bool)
            X_train_clean = X_train_fold
            y_train_clean = y_train_fold
        
        # Box-Cox transformation on clean data
        optimal_lambda = find_optimal_box_cox_lambda(y_train_clean)
        y_train_transformed = box_cox_transform(y_train_clean, optimal_lambda)
        y_val_transformed = box_cox_transform(y_val_original, optimal_lambda)
        
        # Train on clean data
        model = CatBoostRegressor(cat_features=categorical_feature_indices, **params)
        model.fit(X_train_clean, y_train_transformed, 
                 eval_set=(X_val_fold, y_val_transformed), verbose=False)
        
        # Predict on validation (which includes all data)
        pred_transformed = model.predict(X_val_fold)
        rmse_transformed = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores_transformed.append(rmse_transformed)
        
        n_removed = outlier_mask.sum()
        print(f"  Fold {fold}: removed {n_removed} outliers, RMSE = {rmse_transformed:.4f}")
    
    return cv_scores_transformed

def test_baseline_no_outlier_removal(train, test):
    """Test baseline without outlier removal"""
    print("\n" + "=" * 50)
    print("BASELINE: NO OUTLIER REMOVAL")
    print("=" * 50)
    
    cv_scores = test_outlier_method(train, test, method=None, contamination=0)
    baseline_cv = np.mean(cv_scores)
    
    print(f"Baseline (no outlier removal) CV: {baseline_cv:.4f} ± {np.std(cv_scores):.4f}")
    
    return baseline_cv

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Test baseline first
    baseline_cv = test_baseline_no_outlier_removal(train, test)
    
    # Test outlier removal methods
    results = test_outlier_removal_methods(train, test)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("OUTLIER REMOVAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<25} {'CV RMSE':<15} {'Improvement'}")
    print("-" * 60)
    print(f"{'Baseline (no removal)':<25} {baseline_cv:.4f}       {'-'}")
    
    for method_name, cv_scores in results.items():
        cv_mean = np.mean(cv_scores)
        improvement = (baseline_cv - cv_mean) / baseline_cv * 100
        print(f"{method_name:<25} {cv_mean:.4f}       {improvement:+.2f}%")
    
    # Find best method
    best_method = min(results.keys(), key=lambda k: np.mean(results[k]))
    best_score = np.mean(results[best_method])
    
    print(f"\nBest method: {best_method}")
    print(f"Best CV: {best_score:.4f}")
    print(f"Improvement over baseline: {(baseline_cv - best_score) / baseline_cv * 100:+.2f}%")

if __name__ == "__main__":
    main()