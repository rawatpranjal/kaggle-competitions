#!/usr/bin/env python3
"""
CV VALIDATION CHECK
==================
Analyze if there was a CV methodology error in the winsorization approach.
"""

import pandas as pd
import numpy as np
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

def wrong_cv_approach(train, test):
    """WRONG: Apply winsorization to full dataset before CV"""
    print("=" * 60)
    print("WRONG CV APPROACH (What we did)")
    print("=" * 60)
    
    # Apply winsorization to full training set FIRST
    train_winsorized = train.copy()
    y = train['SalePrice']
    
    lower_bound = np.percentile(y, 5)
    upper_bound = np.percentile(y, 95)
    
    print(f"Winsorization bounds calculated on FULL dataset:")
    print(f"  Lower: ${lower_bound:,.0f}")
    print(f"  Upper: ${upper_bound:,.0f}")
    
    train_winsorized.loc[y < lower_bound, 'SalePrice'] = lower_bound
    train_winsorized.loc[y > upper_bound, 'SalePrice'] = upper_bound
    
    winsorized_count = ((y < lower_bound) | (y > upper_bound)).sum()
    print(f"  Winsorized: {winsorized_count} samples")
    
    # Now do CV on the winsorized dataset
    X_train, X_test, cat_features = prepare_features_for_catboost(train_winsorized, test)
    y_train = np.log1p(train_winsorized['SalePrice'])
    
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
    
    print(f"\nCross-validation (WRONG approach):")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        cv_scores.append(rmse)
        
        print(f"  Fold {fold}: RMSE = {rmse:.4f}")
    
    wrong_cv_mean = np.mean(cv_scores)
    print(f"\nWRONG CV Result: {wrong_cv_mean:.4f} ± {np.std(cv_scores):.4f}")
    
    return wrong_cv_mean

def correct_cv_approach(train, test):
    """CORRECT: Apply winsorization within each CV fold"""
    print("\n" + "=" * 60)
    print("CORRECT CV APPROACH (What we should have done)")
    print("=" * 60)
    
    # Prepare features without winsorization first
    X_train_orig, X_test, cat_features = prepare_features_for_catboost(train, test)
    y_train_orig = np.log1p(train['SalePrice'])
    
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
    
    print(f"Cross-validation (CORRECT approach - winsorize within each fold):")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_orig), 1):
        # Split the original data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Apply winsorization ONLY to training fold
        y_train_fold = train_fold['SalePrice']
        lower_bound = np.percentile(y_train_fold, 5)
        upper_bound = np.percentile(y_train_fold, 95)
        
        # Winsorize training fold
        train_fold_wins = train_fold.copy()
        train_fold_wins.loc[y_train_fold < lower_bound, 'SalePrice'] = lower_bound
        train_fold_wins.loc[y_train_fold > upper_bound, 'SalePrice'] = upper_bound
        
        # Prepare features for this fold
        X_train_fold, _, _ = prepare_features_for_catboost(train_fold_wins, test)
        X_val_fold, _, _ = prepare_features_for_catboost(val_fold, test)
        
        y_train_fold_log = np.log1p(train_fold_wins['SalePrice'])
        y_val_fold_log = np.log1p(val_fold['SalePrice'])  # Validation NEVER winsorized
        
        # Train model on winsorized training data
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_train_fold, y_train_fold_log, eval_set=(X_val_fold, y_val_fold_log), verbose=False)
        
        # Predict on original (non-winsorized) validation data
        pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold_log, pred))
        cv_scores.append(rmse)
        
        winsorized_in_fold = ((y_train_fold < lower_bound) | (y_train_fold > upper_bound)).sum()
        
        print(f"  Fold {fold}: RMSE = {rmse:.4f} (winsorized {winsorized_in_fold} in training)")
    
    correct_cv_mean = np.mean(cv_scores)
    print(f"\nCORRECT CV Result: {correct_cv_mean:.4f} ± {np.std(cv_scores):.4f}")
    
    return correct_cv_mean

def analyze_cv_error():
    """Analyze the impact of the CV methodology error"""
    print("\n" + "=" * 60)
    print("CV METHODOLOGY ANALYSIS")
    print("=" * 60)
    
    print("What we did WRONG:")
    print("1. Applied winsorization to FULL training dataset")
    print("2. Then did cross-validation on winsorized data")
    print("3. This creates DATA LEAKAGE:")
    print("   - Validation folds 'saw' the full dataset during winsorization")
    print("   - Outlier bounds were calculated using validation data")
    print("   - CV score became overly optimistic")
    
    print("\nWhat we SHOULD have done:")
    print("1. For each CV fold:")
    print("   - Calculate winsorization bounds ONLY on training fold")
    print("   - Apply winsorization ONLY to training fold") 
    print("   - Validate on ORIGINAL (non-winsorized) validation fold")
    print("2. This prevents data leakage")
    print("3. CV score would be more realistic")
    
    print("\nWhy our leaderboard score was worse:")
    print("- Test set has different price distribution than training set")
    print("- Our winsorization bounds were overfit to training data")
    print("- Model learned to predict within [88k, 326k] range")
    print("- Test set may have different outlier patterns")

def main():
    print("=" * 60)
    print("CV VALIDATION CHECK")
    print("=" * 60)
    
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Show both approaches
    wrong_cv = wrong_cv_approach(train, test)
    correct_cv = correct_cv_approach(train, test)
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Wrong CV (what we did):   {wrong_cv:.4f}")
    print(f"Correct CV (what we should do): {correct_cv:.4f}")
    print(f"Difference: {abs(wrong_cv - correct_cv):.4f}")
    print(f"Our CV was optimistic by: {((correct_cv - wrong_cv) / correct_cv * 100):+.1f}%")
    
    print("\nLeaderboard vs CV comparison:")
    print(f"Our CV score: 0.0997")
    print(f"Leaderboard:  0.14978")
    print(f"Gap: {((0.14978 - 0.0997) / 0.0997 * 100):+.1f}% (CV too optimistic)")

if __name__ == "__main__":
    main()