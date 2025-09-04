#!/usr/bin/env python3
"""
MINIMAL CATBOOST TUNING
=======================
Test just a few key parameter combinations quickly.
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

def prepare_data_fast(train, test):
    """Very fast data preparation"""
    # Combine datasets
    all_data = pd.concat([train, test], ignore_index=True, sort=False)
    
    # Fill numeric NaNs with median
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna(all_data[col].median())
    
    # Fill categorical NaNs and convert to string
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna('Unknown')
        all_data[col] = all_data[col].astype(str)
    
    # Split back
    train_processed = all_data.iloc[:len(train)].reset_index(drop=True)
    test_processed = all_data.iloc[len(train):].reset_index(drop=True)
    
    # Get features
    feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice']]
    X_train = train_processed[feature_cols]
    X_test = test_processed[feature_cols]
    
    # Categorical indices
    categorical_feature_indices = []
    for i, col in enumerate(X_train.columns):
        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'string':
            categorical_feature_indices.append(i)
    
    return X_train, X_test, categorical_feature_indices

def box_cox_transform(y, lam):
    """Apply Box-Cox transformation"""
    y_positive = y + 1
    if abs(lam) < 1e-6:
        return np.log(y_positive)
    else:
        return (np.power(y_positive, lam) - 1) / lam

def test_config(train, test, params, config_name):
    """Test a single configuration"""
    print(f"Testing {config_name}:")
    
    X_train, X_test, cat_features = prepare_data_fast(train, test)
    
    # Use single fold for speed
    train_size = int(0.8 * len(train))
    train_idx = np.arange(train_size)
    val_idx = np.arange(train_size, len(train))
    
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr_orig = train.iloc[train_idx]['SalePrice']
    y_val_orig = train.iloc[val_idx]['SalePrice']
    
    # Box-Cox transform
    y_positive = y_tr_orig + 1
    _, fitted_lambda = stats.boxcox(y_positive)
    
    y_tr = box_cox_transform(y_tr_orig, fitted_lambda)
    y_val = box_cox_transform(y_val_orig, fitted_lambda)
    
    # Train and evaluate
    model = CatBoostRegressor(cat_features=cat_features, **params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    
    print(f"  {config_name}: RMSE = {rmse:.4f}")
    return rmse

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    print("\n" + "=" * 40)
    print("MINIMAL CATBOOST TUNING")
    print("=" * 40)
    
    # Test configurations
    configs = {
        'baseline': {
            'objective': 'RMSE',
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'iterations': 500,
            'random_state': 42,
            'verbose': False
        },
        
        'high_reg': {
            'objective': 'RMSE',
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 8,        # Higher regularization
            'iterations': 500,
            'random_state': 42,
            'verbose': False
        },
        
        'deeper': {
            'objective': 'RMSE',
            'learning_rate': 0.03,   # Lower LR for deeper trees
            'depth': 8,              # Deeper
            'l2_leaf_reg': 5,
            'iterations': 700,
            'random_state': 42,
            'verbose': False
        },
        
        'conservative': {
            'objective': 'RMSE',
            'learning_rate': 0.02,   # Very conservative
            'depth': 5,              # Shallow
            'l2_leaf_reg': 10,       # High reg
            'iterations': 800,
            'random_state': 42,
            'verbose': False
        }
    }
    
    results = {}
    for config_name, params in configs.items():
        results[config_name] = test_config(train, test, params, config_name)
    
    # Results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    
    baseline_score = results['baseline']
    
    for config_name, score in results.items():
        if config_name == 'baseline':
            print(f"{config_name:<12}: {score:.4f} (baseline)")
        else:
            improvement = (baseline_score - score) / baseline_score * 100
            print(f"{config_name:<12}: {score:.4f} ({improvement:+.2f}%)")
    
    best_config = min(results.keys(), key=lambda k: results[k])
    best_score = results[best_config]
    
    print(f"\nBest: {best_config} ({best_score:.4f})")
    
    if best_score < baseline_score:
        improvement = (baseline_score - best_score) / baseline_score * 100
        print(f"Improvement: {improvement:+.2f}%")
        print("✓ Tuning helped!")
    else:
        print("✗ Baseline was best")

if __name__ == "__main__":
    main()