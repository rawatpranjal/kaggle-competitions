#!/usr/bin/env python3
"""
CATBOOST WITH WINSORIZATION
===========================
Test CatBoost full features with 5th-95th percentile winsorization.
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

def apply_winsorization(train, percentiles=(5, 95)):
    """Apply 5th-95th percentile winsorization"""
    train_winsorized = train.copy()
    y = train['SalePrice']
    
    lower_pct, upper_pct = percentiles
    lower_bound = np.percentile(y, lower_pct)
    upper_bound = np.percentile(y, upper_pct)
    
    print(f"Winsorization bounds:")
    print(f"  Lower ({lower_pct}th): ${lower_bound:,.0f}")
    print(f"  Upper ({upper_pct}th): ${upper_bound:,.0f}")
    
    # Apply winsorization
    original_below = (y < lower_bound).sum()
    original_above = (y > upper_bound).sum()
    
    train_winsorized.loc[y < lower_bound, 'SalePrice'] = lower_bound
    train_winsorized.loc[y > upper_bound, 'SalePrice'] = upper_bound
    
    print(f"  Houses winsorized: {original_below + original_above} ({(original_below + original_above)/len(train):.1%})")
    
    return train_winsorized

def prepare_features_for_catboost(train, test):
    """Prepare features for CatBoost with categorical handling"""
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
    
    # Split back
    train_processed = all_data.iloc[:len(train)].reset_index(drop=True)
    test_processed = all_data.iloc[len(train):].reset_index(drop=True)
    
    # Select all features except Id and SalePrice
    feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice']]
    X_train = train_processed[feature_cols]
    X_test = test_processed[feature_cols]
    
    # Identify categorical features for CatBoost
    categorical_feature_indices = []
    for i, col in enumerate(X_train.columns):
        if X_train[col].dtype == 'object':
            categorical_feature_indices.append(i)
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Categorical features: {len(categorical_feature_indices)}")
    
    return X_train, X_test, categorical_feature_indices

def run_catboost_cv_comparison(train_original, train_winsorized, test):
    """Compare CatBoost performance with and without winsorization"""
    print("=" * 70)
    print("CATBOOST WINSORIZATION COMPARISON")
    print("=" * 70)
    
    # Prepare features for both datasets
    X_train_orig, X_test_orig, cat_features_orig = prepare_features_for_catboost(train_original, test)
    X_train_wins, X_test_wins, cat_features_wins = prepare_features_for_catboost(train_winsorized, test)
    
    y_orig = np.log1p(train_original['SalePrice'])
    y_wins = np.log1p(train_winsorized['SalePrice'])
    
    print(f"Original target std: {y_orig.std():.4f}")
    print(f"Winsorized target std: {y_wins.std():.4f}")
    
    # CatBoost parameters (best from previous analysis)
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
    
    # Original data
    print(f"\nOriginal data cross-validation:")
    orig_scores = []
    orig_models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_orig), 1):
        X_tr, X_val = X_train_orig.iloc[train_idx], X_train_orig.iloc[val_idx]
        y_tr, y_val = y_orig.iloc[train_idx], y_orig.iloc[val_idx]
        
        model = CatBoostRegressor(cat_features=cat_features_orig, **params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        orig_scores.append(rmse)
        orig_models.append(model)
        
        print(f"  Fold {fold}: RMSE = {rmse:.4f} (best iter: {model.get_best_iteration()})")
    
    orig_mean = np.mean(orig_scores)
    orig_std = np.std(orig_scores)
    
    print(f"Original CV: {orig_mean:.4f} ± {orig_std:.4f}")
    
    # Winsorized data
    print(f"\nWinsorized data cross-validation:")
    wins_scores = []
    wins_models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_wins), 1):
        X_tr, X_val = X_train_wins.iloc[train_idx], X_train_wins.iloc[val_idx]
        y_tr, y_val = y_wins.iloc[train_idx], y_wins.iloc[val_idx]
        
        model = CatBoostRegressor(cat_features=cat_features_wins, **params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        wins_scores.append(rmse)
        wins_models.append(model)
        
        print(f"  Fold {fold}: RMSE = {rmse:.4f} (best iter: {model.get_best_iteration()})")
    
    wins_mean = np.mean(wins_scores)
    wins_std = np.std(wins_scores)
    
    print(f"Winsorized CV: {wins_mean:.4f} ± {wins_std:.4f}")
    
    # Calculate improvement
    improvement = (orig_mean - wins_mean) / orig_mean * 100
    
    print(f"\n" + "=" * 50)
    print("CATBOOST WINSORIZATION RESULTS")
    print("=" * 50)
    print(f"Original model:   {orig_mean:.4f} ± {orig_std:.4f}")
    print(f"Winsorized model: {wins_mean:.4f} ± {wins_std:.4f}")
    print(f"Improvement:      {improvement:+.2f}%")
    
    # Compare with previous results
    print(f"\n" + "=" * 50)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("=" * 50)
    print("Previous results:")
    print(f"  CatBoost all features (original):     0.1260 ± 0.0169")
    print(f"  LightGBM all features (original):     0.1286 ± 0.0180") 
    print(f"  LightGBM winsorized (5-95th):         0.1054 ± 0.0143")
    print(f"Current:")
    print(f"  CatBoost all features (winsorized):   {wins_mean:.4f} ± {wins_std:.4f}")
    
    if wins_mean < 0.1260:
        print(f"\n🎉 NEW BEST MODEL! Improvement over previous best CatBoost: {((0.1260 - wins_mean) / 0.1260 * 100):+.2f}%")
    
    return wins_models, X_test_wins, wins_mean

def create_submission(models, X_test, test_ids, filename):
    """Create submission using ensemble of CV models"""
    print(f"\n" + "=" * 50)
    print("CREATING CATBOOST WINSORIZED SUBMISSION")
    print("=" * 50)
    
    # Average predictions across CV models
    predictions = np.zeros(len(X_test))
    
    for model in models:
        pred = model.predict(X_test)
        predictions += pred
    
    predictions /= len(models)
    
    # Inverse log transform
    final_predictions = np.expm1(predictions)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': final_predictions
    })
    
    submission.to_csv(f'submissions/{filename}', index=False)
    
    print(f"Submission created:")
    print(f"  File: submissions/{filename}")
    print(f"  Samples: {len(submission)}")
    print(f"  Price range: ${final_predictions.min():,.0f} - ${final_predictions.max():,.0f}")
    print(f"  Price median: ${np.median(final_predictions):,.0f}")
    
    return submission

def main():
    print("=" * 60)
    print("CATBOOST WITH WINSORIZATION")
    print("=" * 60)
    
    # Load data
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Apply 5th-95th percentile winsorization
    train_winsorized = apply_winsorization(train, (5, 95))
    
    # Compare CatBoost performance
    models, X_test, cv_score = run_catboost_cv_comparison(train, train_winsorized, test)
    
    # Create submission with winsorized CatBoost
    test_ids = test['Id'].values
    submission = create_submission(models, X_test, test_ids, 'catboost_winsorized_5_95.csv')
    
    print(f"\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"CatBoost + Winsorization CV: {cv_score:.4f}")
    print("This should be our best performing model!")

if __name__ == "__main__":
    main()