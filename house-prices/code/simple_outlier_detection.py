#!/usr/bin/env python3
"""
SIMPLE MULTIVARIATE OUTLIER DETECTION
=====================================
Fast multivariate outlier detection using Isolation Forest.
"""

import pandas as pd
import numpy as np
from scipy import stats
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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
    """Comprehensive feature engineering (abbreviated for speed)"""
    df_eng = df.copy()
    
    # Key engineered features only
    df_eng['LivingAreaEfficiency'] = df_eng['GrLivArea'] / df_eng['LotArea']
    df_eng['HouseAge'] = df_eng['YrSold'] - df_eng['YearBuilt']
    df_eng['OverallScore'] = df_eng['OverallQual'] * df_eng['OverallCond']
    df_eng['TotalFinishedSF'] = df_eng['GrLivArea'] + df_eng['BsmtFinSF1'] + df_eng['BsmtFinSF2']
    df_eng['TotalBathrooms'] = (df_eng['FullBath'] + 0.5 * df_eng['HalfBath'] + 
                               df_eng['BsmtFullBath'] + 0.5 * df_eng['BsmtHalfBath'])
    
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    df_eng['TotalPorchArea'] = df_eng[porch_cols].sum(axis=1)
    
    # Premium features
    df_eng['HasPool'] = (df_eng['PoolArea'] > 0).astype(int)
    df_eng['HasFireplace'] = (df_eng['Fireplaces'] > 0).astype(int)
    df_eng['HasCentralAir'] = (df_eng['CentralAir'] == 'Y').astype(int)
    
    # Neighborhood stats
    if is_train and 'SalePrice' in df_eng.columns:
        neighborhood_stats = {}
        neighborhood_stats['median_price'] = df_eng.groupby('Neighborhood')['SalePrice'].median()
        
    if neighborhood_stats is not None:
        df_eng['NeighborhoodMedianPrice'] = df_eng['Neighborhood'].map(neighborhood_stats['median_price'])
    
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

def detect_outliers_isolation_forest(X_train, contamination=0.05):
    """Detect outliers using Isolation Forest"""
    # Use only numeric features for outlier detection
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_numeric = X_train[numeric_cols].copy()
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Isolation Forest
    detector = IsolationForest(contamination=contamination, random_state=42, n_jobs=1)
    outlier_pred = detector.fit_predict(X_scaled)
    outlier_mask = outlier_pred == -1
    
    return outlier_mask

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

def test_outlier_contamination_levels(train, test):
    """Test different contamination levels"""
    print("=" * 60)
    print("ISOLATION FOREST CONTAMINATION LEVELS")
    print("=" * 60)
    
    contamination_levels = [0.02, 0.03, 0.05, 0.07, 0.10]
    results = {}
    
    for contamination in contamination_levels:
        print(f"\nTesting contamination = {contamination*100:.1f}%")
        cv_scores = test_outlier_removal(train, test, contamination)
        cv_mean = np.mean(cv_scores)
        results[contamination] = cv_mean
        print(f"CV RMSE: {cv_mean:.4f} ± {np.std(cv_scores):.4f}")
    
    return results

def test_outlier_removal(train, test, contamination=0.05):
    """Test outlier removal with specific contamination level"""
    params = {
        'objective': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'iterations': 800,  # Reduced for speed
        'early_stopping_rounds': 50,
        'random_state': 42,
        'verbose': False,
        'use_best_model': True
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_transformed = []
    outlier_counts = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train), 1):
        # Split original data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Apply feature engineering
        X_train_fold, X_test_fold, cat_features, neighborhood_stats = prepare_features_for_catboost(train_fold, test)
        X_val_fold, _, _, _ = prepare_features_for_catboost(val_fold, test, neighborhood_stats)
        
        y_train_fold = train_fold['SalePrice']
        y_val_original = val_fold['SalePrice']
        
        if contamination > 0:
            # Detect outliers
            outlier_mask = detect_outliers_isolation_forest(X_train_fold, contamination)
            
            # Remove outliers
            X_train_clean = X_train_fold.loc[~outlier_mask].reset_index(drop=True)
            y_train_clean = y_train_fold.loc[~outlier_mask].reset_index(drop=True)
            
            n_removed = outlier_mask.sum()
            outlier_counts.append(n_removed)
            
            if fold == 1:
                outlier_indices = np.where(outlier_mask)[0]
                print(f"  Outlier prices: {[f'${price:,.0f}' for price in y_train_fold.iloc[outlier_indices][:5]]}")
        else:
            # No outlier removal
            X_train_clean = X_train_fold
            y_train_clean = y_train_fold
            n_removed = 0
            outlier_counts.append(0)
        
        # Box-Cox transformation
        optimal_lambda = find_optimal_box_cox_lambda(y_train_clean)
        y_train_transformed = box_cox_transform(y_train_clean, optimal_lambda)
        y_val_transformed = box_cox_transform(y_val_original, optimal_lambda)
        
        # Train model
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_train_clean, y_train_transformed, 
                 eval_set=(X_val_fold, y_val_transformed), verbose=False)
        
        # Predict
        pred_transformed = model.predict(X_val_fold)
        rmse_transformed = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores_transformed.append(rmse_transformed)
        
        print(f"  Fold {fold}: removed {n_removed} outliers, RMSE = {rmse_transformed:.4f}")
    
    avg_outliers_removed = np.mean(outlier_counts)
    print(f"  Average outliers removed per fold: {avg_outliers_removed:.1f}")
    
    return cv_scores_transformed

def test_baseline(train, test):
    """Test baseline without outlier removal"""
    print("\nBASELINE (No outlier removal):")
    cv_scores = test_outlier_removal(train, test, contamination=0)
    baseline_cv = np.mean(cv_scores)
    print(f"Baseline CV: {baseline_cv:.4f} ± {np.std(cv_scores):.4f}")
    return baseline_cv

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Test baseline
    baseline_cv = test_baseline(train, test)
    
    # Test different contamination levels
    results = test_outlier_contamination_levels(train, test)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("OUTLIER REMOVAL RESULTS")
    print("=" * 60)
    print(f"{'Contamination':<15} {'CV RMSE':<10} {'Improvement'}")
    print("-" * 40)
    print(f"{'0% (baseline)':<15} {baseline_cv:.4f}     {'-'}")
    
    for contamination, cv_mean in results.items():
        improvement = (baseline_cv - cv_mean) / baseline_cv * 100
        print(f"{contamination*100:.1f}%            {cv_mean:.4f}     {improvement:+.2f}%")
    
    # Find best contamination level
    best_contamination = min(results.keys(), key=lambda k: results[k])
    best_score = results[best_contamination]
    
    print(f"\nBest contamination level: {best_contamination*100:.1f}%")
    print(f"Best CV: {best_score:.4f}")
    print(f"Improvement over baseline: {(baseline_cv - best_score) / baseline_cv * 100:+.2f}%")
    
    if best_score < baseline_cv:
        print("✓ Outlier removal improves performance!")
    else:
        print("✗ Outlier removal does not improve performance")

if __name__ == "__main__":
    main()