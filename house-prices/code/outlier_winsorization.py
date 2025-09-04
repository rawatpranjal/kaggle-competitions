#!/usr/bin/env python3
"""
HOUSE PRICES OUTLIER WINSORIZATION
==================================
Detect and winsorize extreme house prices, then compare model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train and test data"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def analyze_price_distribution(train):
    """Analyze the distribution of house prices"""
    print("=" * 60)
    print("HOUSE PRICE DISTRIBUTION ANALYSIS")
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
    
    # Calculate percentiles
    percentiles = [1, 5, 10, 90, 95, 99]
    print(f"\nPrice Percentiles:")
    for p in percentiles:
        value = np.percentile(y, p)
        print(f"  {p:2d}th percentile: ${value:8,.0f}")
    
    return y, log_y

def detect_outliers_multiple_methods(train):
    """Detect outliers using multiple methods"""
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION METHODS")
    print("=" * 60)
    
    y = train['SalePrice']
    log_y = np.log1p(y)
    
    outlier_methods = {}
    
    # Method 1: Z-score on log prices
    z_scores = np.abs(stats.zscore(log_y))
    z_outliers = z_scores > 3  # Standard 3-sigma rule
    outlier_methods['z_score'] = z_outliers
    
    print(f"1. Z-Score Method (|z| > 3 on log prices):")
    print(f"   Outliers detected: {z_outliers.sum()}")
    print(f"   Outlier rate: {z_outliers.mean():.1%}")
    
    # Method 2: IQR method on log prices
    Q1 = np.percentile(log_y, 25)
    Q3 = np.percentile(log_y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = (log_y < lower_bound) | (log_y > upper_bound)
    outlier_methods['iqr'] = iqr_outliers
    
    print(f"\n2. IQR Method (1.5*IQR on log prices):")
    print(f"   Lower bound: {np.exp(lower_bound)-1:,.0f}")
    print(f"   Upper bound: {np.exp(upper_bound)-1:,.0f}")
    print(f"   Outliers detected: {iqr_outliers.sum()}")
    print(f"   Outlier rate: {iqr_outliers.mean():.1%}")
    
    # Method 3: Percentile-based (1st and 99th percentiles)
    p1 = np.percentile(y, 1)
    p99 = np.percentile(y, 99)
    percentile_outliers = (y < p1) | (y > p99)
    outlier_methods['percentile'] = percentile_outliers
    
    print(f"\n3. Percentile Method (< 1st or > 99th percentile):")
    print(f"   Lower bound: ${p1:,.0f}")
    print(f"   Upper bound: ${p99:,.0f}")
    print(f"   Outliers detected: {percentile_outliers.sum()}")
    print(f"   Outlier rate: {percentile_outliers.mean():.1%}")
    
    # Method 4: Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    isolation_outliers = iso_forest.fit_predict(log_y.values.reshape(-1, 1)) == -1
    outlier_methods['isolation'] = isolation_outliers
    
    print(f"\n4. Isolation Forest (contamination=0.05):")
    print(f"   Outliers detected: {isolation_outliers.sum()}")
    print(f"   Outlier rate: {isolation_outliers.mean():.1%}")
    
    # Show overlap between methods
    print(f"\n" + "=" * 40)
    print("OUTLIER METHOD OVERLAP")
    print("=" * 40)
    
    methods = list(outlier_methods.keys())
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            overlap = (outlier_methods[method1] & outlier_methods[method2]).sum()
            total_outliers = (outlier_methods[method1] | outlier_methods[method2]).sum()
            if total_outliers > 0:
                overlap_rate = overlap / total_outliers
                print(f"{method1} ∩ {method2}: {overlap}/{total_outliers} ({overlap_rate:.1%})")
    
    return outlier_methods

def apply_winsorization(train, method='iqr', percentiles=(1, 99)):
    """Apply winsorization to the training data"""
    print(f"\n" + "=" * 60)
    print(f"APPLYING WINSORIZATION")
    print("=" * 60)
    
    train_winsorized = train.copy()
    y = train['SalePrice']
    
    if method == 'percentile':
        lower_pct, upper_pct = percentiles
        lower_bound = np.percentile(y, lower_pct)
        upper_bound = np.percentile(y, upper_pct)
        
        print(f"Percentile winsorization ({lower_pct}th - {upper_pct}th):")
        print(f"  Lower bound: ${lower_bound:,.0f}")
        print(f"  Upper bound: ${upper_bound:,.0f}")
        
    elif method == 'iqr':
        log_y = np.log1p(y)
        Q1 = np.percentile(log_y, 25)
        Q3 = np.percentile(log_y, 75)
        IQR = Q3 - Q1
        lower_bound_log = Q1 - 1.5 * IQR
        upper_bound_log = Q3 + 1.5 * IQR
        
        lower_bound = np.exp(lower_bound_log) - 1
        upper_bound = np.exp(upper_bound_log) - 1
        
        print(f"IQR winsorization (1.5*IQR on log scale):")
        print(f"  Lower bound: ${lower_bound:,.0f}")
        print(f"  Upper bound: ${upper_bound:,.0f}")
    
    # Apply winsorization
    original_below = (y < lower_bound).sum()
    original_above = (y > upper_bound).sum()
    
    train_winsorized.loc[y < lower_bound, 'SalePrice'] = lower_bound
    train_winsorized.loc[y > upper_bound, 'SalePrice'] = upper_bound
    
    print(f"\nWinsorization applied:")
    print(f"  Houses capped at lower bound: {original_below}")
    print(f"  Houses capped at upper bound: {original_above}")
    print(f"  Total houses winsorized: {original_below + original_above} ({(original_below + original_above)/len(train):.1%})")
    
    # Show impact on distribution
    y_winsorized = train_winsorized['SalePrice']
    log_y_winsorized = np.log1p(y_winsorized)
    
    print(f"\nDistribution after winsorization:")
    print(f"  Original skewness: {np.log1p(y).skew():.3f} → {log_y_winsorized.skew():.3f}")
    print(f"  Original kurtosis: {np.log1p(y).kurtosis():.3f} → {log_y_winsorized.kurtosis():.3f}")
    print(f"  Original std: {np.log1p(y).std():.3f} → {log_y_winsorized.std():.3f}")
    
    return train_winsorized, lower_bound, upper_bound

def prepare_features_simple(train, test):
    """Simple feature preparation for comparison"""
    # Combine datasets for consistent preprocessing
    all_data = pd.concat([train, test], ignore_index=True, sort=False)
    
    # Handle missing values
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna('Unknown', inplace=True)
    
    # Simple label encoding for categoricals
    from sklearn.preprocessing import LabelEncoder
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
    
    # Split back and select features
    train_processed = all_data.iloc[:len(train)].reset_index(drop=True)
    test_processed = all_data.iloc[len(train):].reset_index(drop=True)
    
    feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice']]
    X_train = train_processed[feature_cols]
    X_test = test_processed[feature_cols]
    
    return X_train, X_test

def compare_model_performance(train_original, train_winsorized, test):
    """Compare model performance with and without winsorization"""
    print(f"\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Prepare features for both datasets
    X_train_orig, X_test_orig = prepare_features_simple(train_original, test)
    X_train_wins, X_test_wins = prepare_features_simple(train_winsorized, test)
    
    y_orig = np.log1p(train_original['SalePrice'])
    y_wins = np.log1p(train_winsorized['SalePrice'])
    
    print(f"Feature matrix shape: {X_train_orig.shape}")
    print(f"Original target std: {y_orig.std():.4f}")
    print(f"Winsorized target std: {y_wins.std():.4f}")
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # LightGBM comparison
    print(f"\n{'Model':<25} {'Original RMSE':<15} {'Winsorized RMSE':<17} {'Improvement':<12}")
    print("-" * 75)
    
    results = {}
    
    # LightGBM
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
    
    # Original data
    lgb_scores_orig = []
    for train_idx, val_idx in cv.split(X_train_orig):
        X_tr, X_val = X_train_orig.iloc[train_idx], X_train_orig.iloc[val_idx]
        y_tr, y_val = y_orig.iloc[train_idx], y_orig.iloc[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(lgb_params, train_data, valid_sets=[val_data], 
                         num_boost_round=1000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        lgb_scores_orig.append(rmse)
    
    # Winsorized data
    lgb_scores_wins = []
    for train_idx, val_idx in cv.split(X_train_wins):
        X_tr, X_val = X_train_wins.iloc[train_idx], X_train_wins.iloc[val_idx]
        y_tr, y_val = y_wins.iloc[train_idx], y_wins.iloc[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(lgb_params, train_data, valid_sets=[val_data], 
                         num_boost_round=1000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        lgb_scores_wins.append(rmse)
    
    lgb_orig_mean = np.mean(lgb_scores_orig)
    lgb_wins_mean = np.mean(lgb_scores_wins)
    lgb_improvement = (lgb_orig_mean - lgb_wins_mean) / lgb_orig_mean * 100
    
    print(f"{'LightGBM':<25} {lgb_orig_mean:8.4f}±{np.std(lgb_scores_orig):.4f}   {lgb_wins_mean:8.4f}±{np.std(lgb_scores_wins):.4f}   {lgb_improvement:+6.2f}%")
    
    results['lightgbm'] = {
        'original': lgb_orig_mean,
        'winsorized': lgb_wins_mean,
        'improvement': lgb_improvement
    }
    
    return results

def create_final_submission(train_winsorized, test, method_name):
    """Create submission with best winsorized model"""
    print(f"\n" + "=" * 50)
    print(f"CREATING WINSORIZED SUBMISSION")
    print("=" * 50)
    
    # Prepare features
    X_train, X_test = prepare_features_simple(train_winsorized, test)
    y_train = np.log1p(train_winsorized['SalePrice'])
    
    # Train final LightGBM model
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
    
    # Train on full dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(lgb_params, train_data, num_boost_round=500)
    
    # Make predictions
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': test_pred
    })
    
    filename = f'submissions/lightgbm_winsorized_{method_name}.csv'
    submission.to_csv(filename, index=False)
    
    print(f"Winsorized submission created:")
    print(f"  File: {filename}")
    print(f"  Samples: {len(submission)}")
    print(f"  Price range: ${test_pred.min():,.0f} - ${test_pred.max():,.0f}")
    print(f"  Price median: ${np.median(test_pred):,.0f}")
    
    return submission

def main():
    print("=" * 60)
    print("HOUSE PRICES OUTLIER WINSORIZATION")
    print("=" * 60)
    
    # Load data
    train, test = load_data()
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Analyze price distribution
    y, log_y = analyze_price_distribution(train)
    
    # Detect outliers using multiple methods
    outlier_methods = detect_outliers_multiple_methods(train)
    
    # Test different winsorization approaches
    winsorization_methods = [
        ('iqr', None),
        ('percentile', (1, 99)),
        ('percentile', (2, 98)),
        ('percentile', (5, 95))
    ]
    
    best_method = None
    best_improvement = -999
    best_train_wins = None
    
    for method, params in winsorization_methods:
        print(f"\n" + "="*80)
        if method == 'percentile':
            method_name = f"{method}_{params[0]}_{params[1]}"
            train_wins, lower, upper = apply_winsorization(train, method, params)
        else:
            method_name = method
            train_wins, lower, upper = apply_winsorization(train, method)
        
        # Compare model performance
        results = compare_model_performance(train, train_wins, test)
        
        lgb_improvement = results['lightgbm']['improvement']
        
        if lgb_improvement > best_improvement:
            best_improvement = lgb_improvement
            best_method = method_name
            best_train_wins = train_wins
        
        print(f"\n{method_name} Summary:")
        print(f"  LightGBM improvement: {lgb_improvement:+.2f}%")
    
    print(f"\n" + "=" * 60)
    print("BEST WINSORIZATION METHOD")
    print("=" * 60)
    print(f"Best method: {best_method}")
    print(f"Best improvement: {best_improvement:+.2f}%")
    
    if best_improvement > 0:
        print(f"Winsorization helps! Creating submission with {best_method}")
        submission = create_final_submission(best_train_wins, test, best_method)
    else:
        print("Winsorization doesn't improve performance. Original data is better.")
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()