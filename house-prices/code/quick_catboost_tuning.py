#!/usr/bin/env python3
"""
QUICK CATBOOST TUNING
====================
Fast CatBoost parameter tuning focusing on key regularization parameters.
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

def engineer_features_fast(df, neighborhood_stats=None, is_train=True):
    """Fast feature engineering - key features only"""
    df_eng = df.copy()
    
    # Core engineered features
    df_eng['LivingAreaEfficiency'] = df_eng['GrLivArea'] / df_eng['LotArea']
    df_eng['HouseAge'] = df_eng['YrSold'] - df_eng['YearBuilt']
    df_eng['OverallScore'] = df_eng['OverallQual'] * df_eng['OverallCond']
    df_eng['TotalFinishedSF'] = df_eng['GrLivArea'] + df_eng['BsmtFinSF1'] + df_eng['BsmtFinSF2']
    df_eng['TotalBathrooms'] = (df_eng['FullBath'] + 0.5 * df_eng['HalfBath'] + 
                               df_eng['BsmtFullBath'] + 0.5 * df_eng['BsmtHalfBath'])
    
    # Premium features
    df_eng['HasFireplace'] = (df_eng['Fireplaces'] > 0).astype(int)
    df_eng['HasCentralAir'] = (df_eng['CentralAir'] == 'Y').astype(int)
    df_eng['HasGarage'] = (df_eng['GarageArea'] > 0).astype(int)
    
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

def fast_nan_handling(train_data, val_data, test_data):
    """Fast NaN handling"""
    # Combine for consistent preprocessing (faster than within-fold for tuning)
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True, sort=False)
    
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna(all_data[col].median())
    
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna('Unknown')
        all_data[col] = all_data[col].astype(str)
    
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

def test_parameter_variation(train, test, param_name, param_values, base_params):
    """Test variation of a single parameter"""
    print(f"Testing {param_name} variations:")
    
    results = {}
    
    for param_value in param_values:
        params = base_params.copy()
        params[param_name] = param_value
        
        cv_scores = quick_cv_test(train, test, params)
        cv_mean = np.mean(cv_scores)
        results[param_value] = cv_mean
        
        print(f"  {param_name}={param_value}: {cv_mean:.4f}")
    
    return results

def quick_cv_test(train, test, params):
    """Quick CV test with reduced iterations"""
    cv = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced to 3 folds
    cv_scores = []
    
    for train_idx, val_idx in cv.split(train):
        # Split data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Fast feature engineering
        train_fold_eng, neighborhood_stats = engineer_features_fast(train_fold, is_train=True)
        val_fold_eng = engineer_features_fast(val_fold, neighborhood_stats, is_train=False)
        test_eng = engineer_features_fast(test, neighborhood_stats, is_train=False)
        
        # Get features
        feature_cols = [col for col in train_fold_eng.columns if col not in ['Id', 'SalePrice', 'PricePerSqFt']]
        
        train_features = train_fold_eng[feature_cols]
        val_features = val_fold_eng[feature_cols]
        test_features = test_eng[feature_cols]
        
        # Fast NaN handling
        train_filled, val_filled, test_filled = fast_nan_handling(
            train_features, val_features, test_features
        )
        
        # Categorical indices
        categorical_feature_indices = []
        for i, col in enumerate(train_filled.columns):
            if train_filled[col].dtype == 'object' or train_filled[col].dtype.name == 'string':
                categorical_feature_indices.append(i)
        
        # Targets
        y_train_fold = train_fold['SalePrice']
        y_val_original = val_fold['SalePrice']
        
        # Box-Cox
        optimal_lambda = find_optimal_box_cox_lambda(y_train_fold)
        y_train_transformed = box_cox_transform(y_train_fold, optimal_lambda)
        y_val_transformed = box_cox_transform(y_val_original, optimal_lambda)
        
        # Train model
        model = CatBoostRegressor(cat_features=categorical_feature_indices, **params)
        model.fit(train_filled, y_train_transformed, 
                 eval_set=(val_filled, y_val_transformed), verbose=False)
        
        # Predict and score
        pred_transformed = model.predict(val_filled)
        rmse = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores.append(rmse)
    
    return cv_scores

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Base parameters (our current best)
    base_params = {
        'objective': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'iterations': 600,  # Reduced for speed
        'early_stopping_rounds': 50,
        'random_state': 42,
        'verbose': False,
        'use_best_model': True
    }
    
    print("\n" + "=" * 50)
    print("QUICK CATBOOST PARAMETER TUNING")
    print("=" * 50)
    
    # Test baseline
    print("Testing baseline configuration:")
    baseline_scores = quick_cv_test(train, test, base_params)
    baseline_cv = np.mean(baseline_scores)
    print(f"Baseline: {baseline_cv:.4f}")
    
    all_results = {'baseline': baseline_cv}
    
    # 1. Test L2 regularization
    print(f"\n1. Testing L2 regularization (l2_leaf_reg):")
    l2_results = test_parameter_variation(
        train, test, 'l2_leaf_reg', [1, 3, 5, 8, 12], base_params
    )
    all_results.update({f'l2_reg_{k}': v for k, v in l2_results.items()})
    
    # 2. Test learning rate
    print(f"\n2. Testing learning rate:")
    lr_results = test_parameter_variation(
        train, test, 'learning_rate', [0.02, 0.03, 0.05, 0.08, 0.1], base_params
    )
    all_results.update({f'lr_{k}': v for k, v in lr_results.items()})
    
    # 3. Test depth
    print(f"\n3. Testing tree depth:")
    depth_results = test_parameter_variation(
        train, test, 'depth', [4, 5, 6, 7, 8], base_params
    )
    all_results.update({f'depth_{k}': v for k, v in depth_results.items()})
    
    # 4. Test iterations (with best params so far)
    best_l2 = min(l2_results.keys(), key=lambda k: l2_results[k])
    best_lr = min(lr_results.keys(), key=lambda k: lr_results[k])
    best_depth = min(depth_results.keys(), key=lambda k: depth_results[k])
    
    optimized_params = base_params.copy()
    optimized_params['l2_leaf_reg'] = best_l2
    optimized_params['learning_rate'] = best_lr  
    optimized_params['depth'] = best_depth
    
    print(f"\n4. Testing iterations with optimized params:")
    print(f"   Using l2_leaf_reg={best_l2}, learning_rate={best_lr}, depth={best_depth}")
    
    iter_results = test_parameter_variation(
        train, test, 'iterations', [400, 600, 800, 1000, 1200], optimized_params
    )
    all_results.update({f'iter_{k}': v for k, v in iter_results.items()})
    
    # Final comparison
    print("\n" + "=" * 50)
    print("TUNING RESULTS SUMMARY")
    print("=" * 50)
    
    best_config = min(all_results.keys(), key=lambda k: all_results[k])
    best_score = all_results[best_config]
    
    print(f"Baseline CV:     {baseline_cv:.4f}")
    print(f"Best config:     {best_config}")
    print(f"Best CV:         {best_score:.4f}")
    
    improvement = (baseline_cv - best_score) / baseline_cv * 100
    print(f"Improvement:     {improvement:+.2f}%")
    
    if best_score < baseline_cv:
        print("✓ Parameter tuning helped!")
        print(f"\nRecommended parameters:")
        print(f"  l2_leaf_reg: {best_l2}")
        print(f"  learning_rate: {best_lr}")
        print(f"  depth: {best_depth}")
        print(f"  iterations: Use early stopping with sufficient budget")
    else:
        print("✗ Baseline parameters were already optimal")

if __name__ == "__main__":
    main()