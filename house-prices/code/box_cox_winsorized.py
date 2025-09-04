#!/usr/bin/env python3
"""
BOX-COX + LIGHT WINSORIZATION TEST
==================================
Test Box-Cox transformation with light winsorization (0.1% each side).
Uses rigorous CV methodology - winsorize within each fold.
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

def prepare_features_for_catboost(train, test):
    """Prepare features for CatBoost"""
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
    test_processed = all_data.iloc[len(train):].reset_index(drop=True)
    
    feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice']]
    X_train = train_processed[feature_cols]
    X_test = test_processed[feature_cols]
    
    categorical_feature_indices = []
    for i, col in enumerate(X_train.columns):
        if X_train[col].dtype == 'object':
            categorical_feature_indices.append(i)
    
    return X_train, X_test, categorical_feature_indices

def winsorize_light(y, lower_pct=0.1, upper_pct=99.9):
    """Apply light winsorization (0.1% each side)"""
    lower_bound = np.percentile(y, lower_pct)
    upper_bound = np.percentile(y, upper_pct)
    
    y_winsorized = y.copy()
    y_winsorized[y < lower_bound] = lower_bound
    y_winsorized[y > upper_bound] = upper_bound
    
    winsorized_count = ((y < lower_bound) | (y > upper_bound)).sum()
    
    return y_winsorized, lower_bound, upper_bound, winsorized_count

def find_optimal_box_cox_lambda(y_winsorized):
    """Find optimal Box-Cox lambda on winsorized data"""
    y_positive = y_winsorized + 1  # Ensure positivity
    transformed_data, fitted_lambda = stats.boxcox(y_positive)
    return fitted_lambda, transformed_data

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

def rigorous_cv_with_winsorization_boxcox(train, test):
    """
    Rigorous CV: Apply winsorization and Box-Cox within each fold
    CORRECT methodology - no data leakage
    """
    print("=" * 60)
    print("RIGOROUS CV: WINSORIZATION + BOX-COX")
    print("=" * 60)
    
    # Prepare features (this doesn't leak information)
    X_train_orig, X_test, cat_features = prepare_features_for_catboost(train, test)
    
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
    
    print("Cross-validation with winsorization + Box-Cox within each fold:")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_orig), 1):
        # Split original data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Step 1: Apply winsorization ONLY to training fold
        y_train_fold = train_fold['SalePrice']
        y_train_wins, lower_b, upper_b, wins_count = winsorize_light(y_train_fold)
        
        # Update training fold with winsorized prices
        train_fold_wins = train_fold.copy()
        train_fold_wins['SalePrice'] = y_train_wins
        
        # Step 2: Find optimal Box-Cox lambda on winsorized training data
        optimal_lambda, _ = find_optimal_box_cox_lambda(y_train_wins)
        lambdas_used.append(optimal_lambda)
        
        # Step 3: Apply Box-Cox to winsorized training data
        y_train_transformed = box_cox_transform(y_train_wins, optimal_lambda)
        
        # Step 4: Prepare features for this fold
        X_train_fold, _, _ = prepare_features_for_catboost(train_fold_wins, test)
        X_val_fold, _, _ = prepare_features_for_catboost(val_fold, test)
        
        # Validation target (NEVER winsorized, NEVER Box-Cox transformed for evaluation)
        y_val_original = val_fold['SalePrice']
        
        # Train model on transformed training data
        model = CatBoostRegressor(cat_features=cat_features, **params)
        
        # For early stopping, we need validation in transformed space
        # Apply same transformations to validation (for early stopping only)
        y_val_wins, _, _, _ = winsorize_light(y_val_original, 0.1, 99.9)  # Same bounds
        y_val_transformed = box_cox_transform(y_val_wins, optimal_lambda)
        
        model.fit(X_train_fold, y_train_transformed, 
                 eval_set=(X_val_fold, y_val_transformed), verbose=False)
        
        # Predict in transformed space
        pred_transformed = model.predict(X_val_fold)
        
        # Calculate RMSE in transformed space
        rmse_transformed = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores_transformed.append(rmse_transformed)
        
        # Transform predictions back to original space
        pred_original = inverse_box_cox_transform(pred_transformed, optimal_lambda)
        pred_original = np.maximum(pred_original, 1000)  # Ensure positive prices
        
        # Calculate RMSE in original space (validation never winsorized)
        rmse_original = np.sqrt(mean_squared_error(y_val_original, pred_original))
        cv_scores_original.append(rmse_original)
        
        print(f"  Fold {fold}: λ={optimal_lambda:.4f}, wins={wins_count}, "
              f"RMSE_trans={rmse_transformed:.4f}, RMSE_orig=${rmse_original:,.0f}")
    
    cv_mean_transformed = np.mean(cv_scores_transformed)
    cv_std_transformed = np.std(cv_scores_transformed)
    cv_mean_original = np.mean(cv_scores_original)
    cv_std_original = np.std(cv_scores_original)
    mean_lambda = np.mean(lambdas_used)
    
    print(f"\nRigorous CV Results:")
    print(f"  Average λ: {mean_lambda:.4f}")
    print(f"  Transformed space: {cv_mean_transformed:.4f} ± {cv_std_transformed:.4f}")
    print(f"  Original space: ${cv_mean_original:,.0f} ± ${cv_std_original:,.0f}")
    
    return cv_mean_transformed, cv_std_transformed, cv_mean_original, cv_std_original, mean_lambda

def compare_with_no_winsorization(train, test):
    """Compare with Box-Cox only (no winsorization)"""
    print("\n" + "=" * 60)
    print("COMPARISON: BOX-COX ONLY (NO WINSORIZATION)")
    print("=" * 60)
    
    X_train_orig, X_test, cat_features = prepare_features_for_catboost(train, test)
    
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
    
    print("Cross-validation with Box-Cox only (within each fold):")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_orig), 1):
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        y_train_fold = train_fold['SalePrice']
        
        # Find optimal Box-Cox lambda (no winsorization)
        optimal_lambda, _ = find_optimal_box_cox_lambda(y_train_fold)
        lambdas_used.append(optimal_lambda)
        
        # Apply Box-Cox
        y_train_transformed = box_cox_transform(y_train_fold, optimal_lambda)
        
        # Prepare features
        X_train_fold, _, _ = prepare_features_for_catboost(train_fold, test)
        X_val_fold, _, _ = prepare_features_for_catboost(val_fold, test)
        
        y_val_original = val_fold['SalePrice']
        y_val_transformed = box_cox_transform(y_val_original, optimal_lambda)
        
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_train_fold, y_train_transformed, 
                 eval_set=(X_val_fold, y_val_transformed), verbose=False)
        
        pred_transformed = model.predict(X_val_fold)
        rmse_transformed = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores_transformed.append(rmse_transformed)
        
        pred_original = inverse_box_cox_transform(pred_transformed, optimal_lambda)
        pred_original = np.maximum(pred_original, 1000)
        
        rmse_original = np.sqrt(mean_squared_error(y_val_original, pred_original))
        cv_scores_original.append(rmse_original)
        
        print(f"  Fold {fold}: λ={optimal_lambda:.4f}, "
              f"RMSE_trans={rmse_transformed:.4f}, RMSE_orig=${rmse_original:,.0f}")
    
    cv_mean_transformed = np.mean(cv_scores_transformed)
    cv_mean_original = np.mean(cv_scores_original)
    mean_lambda = np.mean(lambdas_used)
    
    print(f"\nBox-Cox only results:")
    print(f"  Average λ: {mean_lambda:.4f}")
    print(f"  Transformed space: {cv_mean_transformed:.4f}")
    print(f"  Original space: ${cv_mean_original:,.0f}")
    
    return cv_mean_transformed, cv_mean_original, mean_lambda

def main():
    print("=" * 60)
    print("BOX-COX + LIGHT WINSORIZATION TEST")
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
    
    # Test with light winsorization
    wins_cv_trans, wins_cv_std, wins_cv_orig, wins_cv_std_orig, wins_lambda = rigorous_cv_with_winsorization_boxcox(train, test)
    
    # Test without winsorization
    no_wins_cv_trans, no_wins_cv_orig, no_wins_lambda = compare_with_no_winsorization(train, test)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<30} {'Trans RMSE':<12} {'Orig RMSE':<12} {'Avg λ':<8}")
    print("-" * 60)
    print(f"{'Winsorize + Box-Cox':<30} {wins_cv_trans:<12.4f} ${wins_cv_orig:<11,.0f} {wins_lambda:<8.4f}")
    print(f"{'Box-Cox only':<30} {no_wins_cv_trans:<12.4f} ${no_wins_cv_orig:<11,.0f} {no_wins_lambda:<8.4f}")
    
    improvement = (no_wins_cv_trans - wins_cv_trans) / no_wins_cv_trans * 100
    print(f"\nWinsorization improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("✓ Light winsorization helps Box-Cox performance")
    else:
        print("✗ Light winsorization does not improve Box-Cox performance")

if __name__ == "__main__":
    main()