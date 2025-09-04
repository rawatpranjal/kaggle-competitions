#!/usr/bin/env python3
"""
LIGHT WINSORIZATION TEST
=======================
Test lighter winsorization (1st-99th percentiles) with correct CV methodology.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
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

def prepare_features_for_lightgbm(train, test):
    """Prepare features for LightGBM with label encoding"""
    all_data = pd.concat([train, test], ignore_index=True, sort=False)
    
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna('Unknown', inplace=True)
    
    # Label encode categorical features
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
    
    train_processed = all_data.iloc[:len(train)].reset_index(drop=True)
    test_processed = all_data.iloc[len(train):].reset_index(drop=True)
    
    feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice']]
    X_train = train_processed[feature_cols]
    X_test = test_processed[feature_cols]
    
    return X_train, X_test

def apply_winsorization_to_fold(train_fold, lower_pct, upper_pct):
    """Apply winsorization to a single fold"""
    train_fold_wins = train_fold.copy()
    y = train_fold['SalePrice']
    
    lower_bound = np.percentile(y, lower_pct)
    upper_bound = np.percentile(y, upper_pct)
    
    train_fold_wins.loc[y < lower_bound, 'SalePrice'] = lower_bound
    train_fold_wins.loc[y > upper_bound, 'SalePrice'] = upper_bound
    
    winsorized_count = ((y < lower_bound) | (y > upper_bound)).sum()
    
    return train_fold_wins, lower_bound, upper_bound, winsorized_count

def correct_cv_catboost(train, test, lower_pct=1, upper_pct=99):
    """Correct CV with CatBoost and winsorization within folds"""
    print(f"\nCatBoost with {lower_pct}-{upper_pct}th percentile winsorization (CORRECT CV):")
    
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
    
    # Get categorical features (same for all folds)
    _, _, cat_features = prepare_features_for_catboost(train, test)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train), 1):
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Apply winsorization ONLY to training fold
        train_fold_wins, lower_bound, upper_bound, winsorized_count = apply_winsorization_to_fold(
            train_fold, lower_pct, upper_pct
        )
        
        # Prepare features for this fold
        X_train_fold, _, _ = prepare_features_for_catboost(train_fold_wins, test)
        X_val_fold, _, _ = prepare_features_for_catboost(val_fold, test)
        
        y_train_fold_log = np.log1p(train_fold_wins['SalePrice'])
        y_val_fold_log = np.log1p(val_fold['SalePrice'])  # Original validation data
        
        # Train model
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_train_fold, y_train_fold_log, eval_set=(X_val_fold, y_val_fold_log), verbose=False)
        
        # Predict on original validation data
        pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold_log, pred))
        cv_scores.append(rmse)
        models.append(model)
        
        print(f"  Fold {fold}: RMSE = {rmse:.4f} (bounds: ${lower_bound:,.0f}-${upper_bound:,.0f}, winsorized {winsorized_count})")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"  CV Result: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return cv_mean, cv_std, models

def correct_cv_lightgbm(train, test, lower_pct=1, upper_pct=99):
    """Correct CV with LightGBM and winsorization within folds"""
    print(f"\nLightGBM with {lower_pct}-{upper_pct}th percentile winsorization (CORRECT CV):")
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train), 1):
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        
        # Apply winsorization ONLY to training fold
        train_fold_wins, lower_bound, upper_bound, winsorized_count = apply_winsorization_to_fold(
            train_fold, lower_pct, upper_pct
        )
        
        # Prepare features for this fold
        X_train_fold, _ = prepare_features_for_lightgbm(train_fold_wins, test)
        X_val_fold, _ = prepare_features_for_lightgbm(val_fold, test)
        
        y_train_fold_log = np.log1p(train_fold_wins['SalePrice'])
        y_val_fold_log = np.log1p(val_fold['SalePrice'])  # Original validation data
        
        # Train LightGBM
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold_log)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold_log, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predict on original validation data
        pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold_log, pred))
        cv_scores.append(rmse)
        models.append(model)
        
        print(f"  Fold {fold}: RMSE = {rmse:.4f} (bounds: ${lower_bound:,.0f}-${upper_bound:,.0f}, winsorized {winsorized_count})")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"  CV Result: {cv_mean:.4f} ± {cv_std:.4f}")
    
    return cv_mean, cv_std, models

def create_submission_with_correct_winsorization(train, test, model_type='catboost', lower_pct=1, upper_pct=99):
    """Create submission with correct winsorization approach"""
    print(f"\nCreating submission with {model_type} ({lower_pct}-{upper_pct}th percentiles):")
    
    # Apply winsorization to full training data for final model
    y = train['SalePrice']
    lower_bound = np.percentile(y, lower_pct)
    upper_bound = np.percentile(y, upper_pct)
    
    train_wins = train.copy()
    train_wins.loc[y < lower_bound, 'SalePrice'] = lower_bound
    train_wins.loc[y > upper_bound, 'SalePrice'] = upper_bound
    
    print(f"Final model winsorization: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
    
    if model_type == 'catboost':
        X_train, X_test, cat_features = prepare_features_for_catboost(train_wins, test)
        y_train = np.log1p(train_wins['SalePrice'])
        
        params = {
            'objective': 'RMSE',
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'iterations': 600,  # Fewer iterations for final model
            'random_seed': 42,
            'verbose': False
        }
        
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_train, y_train)
        
        test_pred_log = model.predict(X_test)
        filename = f'catboost_winsorized_{lower_pct}_{upper_pct}.csv'
        
    else:  # lightgbm
        X_train, X_test = prepare_features_for_lightgbm(train_wins, test)
        y_train = np.log1p(train_wins['SalePrice'])
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(lgb_params, train_data, num_boost_round=400)
        
        test_pred_log = model.predict(X_test)
        filename = f'lightgbm_winsorized_{lower_pct}_{upper_pct}.csv'
    
    # Create submission
    test_pred = np.expm1(test_pred_log)
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': test_pred
    })
    
    submission.to_csv(f'submissions/{filename}', index=False)
    
    print(f"Submission created: {filename}")
    print(f"Price range: ${test_pred.min():,.0f} - ${test_pred.max():,.0f}")
    
    return submission

def main():
    print("=" * 60)
    print("LIGHT WINSORIZATION TEST (CORRECT CV)")
    print("=" * 60)
    
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Test different winsorization levels with CORRECT CV
    winsorization_levels = [
        (1, 99),   # Light winsorization
        (2, 98),   # Moderate winsorization
        (0, 100),  # No winsorization (baseline)
    ]
    
    results = {}
    
    print(f"\n" + "=" * 60)
    print("CORRECT CV COMPARISON")
    print("=" * 60)
    
    # Test CatBoost with different winsorization levels
    print("Testing CatBoost:")
    for lower_pct, upper_pct in winsorization_levels:
        if lower_pct == 0 and upper_pct == 100:
            print(f"\nCatBoost with NO winsorization (baseline):")
            # Use our previous correct CV approach without winsorization
            _, _, cat_features = prepare_features_for_catboost(train, test)
            X_train, X_test = prepare_features_for_catboost(train, test)[:2]
            y_train = np.log1p(train['SalePrice'])
            
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
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = CatBoostRegressor(cat_features=cat_features, **params)
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
                
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                cv_scores.append(rmse)
                
                print(f"  Fold {fold}: RMSE = {rmse:.4f}")
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            print(f"  CV Result: {cv_mean:.4f} ± {cv_std:.4f}")
            results[f'catboost_{lower_pct}_{upper_pct}'] = (cv_mean, cv_std)
        else:
            cv_mean, cv_std, models = correct_cv_catboost(train, test, lower_pct, upper_pct)
            results[f'catboost_{lower_pct}_{upper_pct}'] = (cv_mean, cv_std)
    
    # Test LightGBM with different winsorization levels  
    print(f"\nTesting LightGBM:")
    for lower_pct, upper_pct in winsorization_levels:
        if lower_pct == 0 and upper_pct == 100:
            print(f"\nLightGBM with NO winsorization (baseline):")
            # Baseline without winsorization
            X_train, X_test = prepare_features_for_lightgbm(train, test)
            y_train = np.log1p(train['SalePrice'])
            
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'verbose': -1,
                'random_state': 42
            }
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                train_data = lgb.Dataset(X_tr, label=y_tr)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    lgb_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
                
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                cv_scores.append(rmse)
                
                print(f"  Fold {fold}: RMSE = {rmse:.4f}")
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            print(f"  CV Result: {cv_mean:.4f} ± {cv_std:.4f}")
            results[f'lightgbm_{lower_pct}_{upper_pct}'] = (cv_mean, cv_std)
        else:
            cv_mean, cv_std, models = correct_cv_lightgbm(train, test, lower_pct, upper_pct)
            results[f'lightgbm_{lower_pct}_{upper_pct}'] = (cv_mean, cv_std)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("RESULTS SUMMARY (CORRECT CV)")
    print("=" * 60)
    print(f"{'Model':<25} {'CV RMSE':<12} {'CV Std':<10}")
    print("-" * 50)
    
    for model_name, (cv_mean, cv_std) in results.items():
        print(f"{model_name:<25} {cv_mean:8.4f}     {cv_std:6.4f}")
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k][0])
    best_rmse = results[best_model][0]
    
    print(f"\nBest model: {best_model}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    # Create submission with best approach
    if 'catboost_1_99' in best_model:
        submission = create_submission_with_correct_winsorization(train, test, 'catboost', 1, 99)
    elif 'catboost_2_98' in best_model:
        submission = create_submission_with_correct_winsorization(train, test, 'catboost', 2, 98)
    elif 'lightgbm_1_99' in best_model:
        submission = create_submission_with_correct_winsorization(train, test, 'lightgbm', 1, 99)
    elif 'lightgbm_2_98' in best_model:
        submission = create_submission_with_correct_winsorization(train, test, 'lightgbm', 2, 98)
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Key insights:")
    print("- Used CORRECT CV methodology (no data leakage)")
    print("- Winsorization applied within each fold only")
    print("- Validation on original (non-winsorized) data")
    print("- Results should align better with leaderboard")

if __name__ == "__main__":
    main()