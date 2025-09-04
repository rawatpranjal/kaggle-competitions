#!/usr/bin/env python3
"""
NO LOG TRANSFORM TEST
====================
Test models without log transformation of target variable.
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

def analyze_target_distribution(train):
    """Analyze target distribution with and without log transform"""
    print("=" * 60)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    y = train['SalePrice']
    log_y = np.log1p(y)
    
    print(f"Original SalePrice:")
    print(f"  Mean: ${y.mean():,.0f}")
    print(f"  Median: ${y.median():,.0f}")
    print(f"  Std: ${y.std():,.0f}")
    print(f"  Min: ${y.min():,.0f}")
    print(f"  Max: ${y.max():,.0f}")
    print(f"  Skewness: {y.skew():.3f}")
    print(f"  Kurtosis: {y.kurtosis():.3f}")
    
    print(f"\nLog-transformed SalePrice:")
    print(f"  Mean: {log_y.mean():.3f}")
    print(f"  Std: {log_y.std():.3f}")
    print(f"  Skewness: {log_y.skew():.3f}")
    print(f"  Kurtosis: {log_y.kurtosis():.3f}")
    
    print(f"\nWhy we typically use log transform:")
    print(f"  - Reduces skewness: {y.skew():.3f} → {log_y.skew():.3f}")
    print(f"  - Normalizes distribution for regression")
    print(f"  - Stabilizes variance across price ranges")
    
    return y, log_y

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

def cv_catboost_no_log(train, test):
    """CatBoost without log transformation"""
    print(f"\n" + "=" * 50)
    print("CATBOOST WITHOUT LOG TRANSFORMATION")
    print("=" * 50)
    
    X_train, X_test, cat_features = prepare_features_for_catboost(train, test)
    y_train = train['SalePrice']  # NO log transform
    
    print(f"Target statistics (no log transform):")
    print(f"  Mean: ${y_train.mean():,.0f}")
    print(f"  Std: ${y_train.std():,.0f}")
    print(f"  Range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    
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
    cv_scores_normalized = []
    models = []
    
    print(f"\nCross-validation (no log transform):")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        pred = model.predict(X_val)
        
        # Calculate RMSE on original scale
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        cv_scores.append(rmse)
        
        # Calculate RMSE on log scale for comparison
        rmse_log = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(pred)))
        cv_scores_normalized.append(rmse_log)
        
        models.append(model)
        
        print(f"  Fold {fold}: RMSE = ${rmse:,.0f} (log-scale RMSE = {rmse_log:.4f})")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    cv_mean_log = np.mean(cv_scores_normalized)
    cv_std_log = np.std(cv_scores_normalized)
    
    print(f"\nCV Results (original scale): ${cv_mean:,.0f} ± ${cv_std:,.0f}")
    print(f"CV Results (log scale): {cv_mean_log:.4f} ± {cv_std_log:.4f}")
    
    return models, cv_mean, cv_std, cv_mean_log, cv_std_log

def cv_catboost_with_log(train, test):
    """CatBoost with log transformation (baseline)"""
    print(f"\n" + "=" * 50)
    print("CATBOOST WITH LOG TRANSFORMATION (BASELINE)")
    print("=" * 50)
    
    X_train, X_test, cat_features = prepare_features_for_catboost(train, test)
    y_train = np.log1p(train['SalePrice'])  # WITH log transform
    
    print(f"Target statistics (log transformed):")
    print(f"  Mean: {y_train.mean():.3f}")
    print(f"  Std: {y_train.std():.3f}")
    print(f"  Range: {y_train.min():.3f} - {y_train.max():.3f}")
    
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
    cv_scores_original = []
    models = []
    
    print(f"\nCross-validation (with log transform):")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        pred_log = model.predict(X_val)
        
        # Calculate RMSE on log scale
        rmse_log = np.sqrt(mean_squared_error(y_val, pred_log))
        cv_scores.append(rmse_log)
        
        # Calculate RMSE on original scale
        pred_original = np.expm1(pred_log)
        y_val_original = np.expm1(y_val)
        rmse_original = np.sqrt(mean_squared_error(y_val_original, pred_original))
        cv_scores_original.append(rmse_original)
        
        models.append(model)
        
        print(f"  Fold {fold}: RMSE = {rmse_log:.4f} (original-scale RMSE = ${rmse_original:,.0f})")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    cv_mean_original = np.mean(cv_scores_original)
    cv_std_original = np.std(cv_scores_original)
    
    print(f"\nCV Results (log scale): {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"CV Results (original scale): ${cv_mean_original:,.0f} ± ${cv_std_original:,.0f}")
    
    return models, cv_mean, cv_std, cv_mean_original, cv_std_original

def cv_lightgbm_comparison(train, test):
    """Compare LightGBM with and without log transform"""
    print(f"\n" + "=" * 50)
    print("LIGHTGBM COMPARISON")
    print("=" * 50)
    
    X_train, X_test = prepare_features_for_lightgbm(train, test)
    
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
    
    # Without log transform
    print("LightGBM WITHOUT log transform:")
    y_train_no_log = train['SalePrice']
    cv_scores_no_log = []
    cv_scores_no_log_normalized = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train_no_log.iloc[train_idx], y_train_no_log.iloc[val_idx]
        
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
        rmse_log = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(pred)))
        
        cv_scores_no_log.append(rmse)
        cv_scores_no_log_normalized.append(rmse_log)
        
        print(f"  Fold {fold}: RMSE = ${rmse:,.0f} (log-scale = {rmse_log:.4f})")
    
    no_log_mean = np.mean(cv_scores_no_log)
    no_log_std = np.std(cv_scores_no_log)
    no_log_mean_normalized = np.mean(cv_scores_no_log_normalized)
    
    print(f"  CV Result: ${no_log_mean:,.0f} ± ${no_log_std:,.0f} (log-scale: {no_log_mean_normalized:.4f})")
    
    # With log transform
    print(f"\nLightGBM WITH log transform:")
    y_train_log = np.log1p(train['SalePrice'])
    cv_scores_log = []
    cv_scores_log_original = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        pred_log = model.predict(X_val)
        rmse_log = np.sqrt(mean_squared_error(y_val, pred_log))
        
        pred_original = np.expm1(pred_log)
        y_val_original = np.expm1(y_val)
        rmse_original = np.sqrt(mean_squared_error(y_val_original, pred_original))
        
        cv_scores_log.append(rmse_log)
        cv_scores_log_original.append(rmse_original)
        
        print(f"  Fold {fold}: RMSE = {rmse_log:.4f} (original-scale = ${rmse_original:,.0f})")
    
    log_mean = np.mean(cv_scores_log)
    log_std = np.std(cv_scores_log)
    log_mean_original = np.mean(cv_scores_log_original)
    
    print(f"  CV Result: {log_mean:.4f} ± {log_std:.4f} (original-scale: ${log_mean_original:,.0f})")
    
    return no_log_mean_normalized, log_mean

def create_submission_no_log(train, test):
    """Create submission with no log transform"""
    print(f"\n" + "=" * 40)
    print("CREATING NO-LOG SUBMISSION")
    print("=" * 40)
    
    X_train, X_test, cat_features = prepare_features_for_catboost(train, test)
    y_train = train['SalePrice']  # No log transform
    
    params = {
        'objective': 'RMSE',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'iterations': 600,
        'random_seed': 42,
        'verbose': False
    }
    
    model = CatBoostRegressor(cat_features=cat_features, **params)
    model.fit(X_train, y_train)
    
    # Predict directly in original scale
    test_pred = model.predict(X_test)
    
    # Ensure predictions are positive
    test_pred = np.maximum(test_pred, 1000)  # Minimum $1000
    
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': test_pred
    })
    
    submission.to_csv('submissions/catboost_no_log.csv', index=False)
    
    print(f"Submission created: catboost_no_log.csv")
    print(f"Price range: ${test_pred.min():,.0f} - ${test_pred.max():,.0f}")
    print(f"Price median: ${np.median(test_pred):,.0f}")
    
    return submission

def main():
    print("=" * 60)
    print("NO LOG TRANSFORM TEST")
    print("=" * 60)
    
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Analyze target distribution
    y, log_y = analyze_target_distribution(train)
    
    # CatBoost comparison
    print(f"\n" + "=" * 60)
    print("CATBOOST: LOG vs NO-LOG COMPARISON")
    print("=" * 60)
    
    # No log transform
    models_no_log, cv_no_log_orig, cv_no_log_std_orig, cv_no_log_norm, cv_no_log_std_norm = cv_catboost_no_log(train, test)
    
    # With log transform
    models_log, cv_log_norm, cv_log_std_norm, cv_log_orig, cv_log_std_orig = cv_catboost_with_log(train, test)
    
    # LightGBM comparison
    lgb_no_log_norm, lgb_log_norm = cv_lightgbm_comparison(train, test)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"CatBoost Results (log-scale RMSE for fair comparison):")
    print(f"  With log transform:    {cv_log_norm:.4f} ± {cv_log_std_norm:.4f}")
    print(f"  Without log transform: {cv_no_log_norm:.4f} ± {cv_no_log_std_norm:.4f}")
    print(f"  Difference: {((cv_no_log_norm - cv_log_norm) / cv_log_norm * 100):+.1f}%")
    
    print(f"\nLightGBM Results (log-scale RMSE):")
    print(f"  With log transform:    {lgb_log_norm:.4f}")
    print(f"  Without log transform: {lgb_no_log_norm:.4f}")
    print(f"  Difference: {((lgb_no_log_norm - lgb_log_norm) / lgb_log_norm * 100):+.1f}%")
    
    print(f"\nOriginal Scale RMSE (more interpretable):")
    print(f"  CatBoost with log:     ${cv_log_orig:,.0f}")
    print(f"  CatBoost without log:  ${cv_no_log_orig:,.0f}")
    
    # Determine if no-log is better
    if cv_no_log_norm < cv_log_norm:
        print(f"\n🎉 NO-LOG TRANSFORM WINS for CatBoost!")
        print(f"Improvement: {((cv_log_norm - cv_no_log_norm) / cv_log_norm * 100):+.1f}%")
        
        # Create submission
        submission = create_submission_no_log(train, test)
        
    else:
        print(f"\nLog transform is still better for CatBoost")
        print(f"Log transform advantage: {((cv_no_log_norm - cv_log_norm) / cv_log_norm * 100):+.1f}%")
    
    print(f"\n" + "=" * 50)
    print("KEY INSIGHTS")
    print("=" * 50)
    print("- Tree-based models can handle skewed targets better than linear models")
    print("- Log transform traditionally helps with:")
    print("  * Normalizing residuals")
    print("  * Stabilizing variance")
    print("  * Meeting regression assumptions")
    print("- Modern gradient boosting may not need log transform")
    print("- Test both approaches for optimal performance")

if __name__ == "__main__":
    main()