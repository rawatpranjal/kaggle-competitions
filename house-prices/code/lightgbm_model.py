#!/usr/bin/env python3
"""
HOUSE PRICES LIGHTGBM MODEL
===========================
Run LightGBM on selected features from correlation analysis.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train and test data"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def load_selected_features():
    """Load selected features from correlation analysis"""
    selected_df = pd.read_csv('selected_features.csv')
    return selected_df['feature'].tolist()

def prepare_features(train, test, feature_list=None):
    """Prepare features for modeling"""
    print("Preparing features...")
    
    # Combine datasets for consistent preprocessing
    all_data = pd.concat([train, test], ignore_index=True, sort=False)
    
    # Handle missing values
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    # Fill categorical missing values with mode or 'Unknown'
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            mode_val = all_data[col].mode()
            if len(mode_val) > 0:
                all_data[col].fillna(mode_val[0], inplace=True)
            else:
                all_data[col].fillna('Unknown', inplace=True)
    
    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        encoders[col] = le
    
    # Split back
    train_processed = all_data.iloc[:len(train)].reset_index(drop=True)
    test_processed = all_data.iloc[len(train):].reset_index(drop=True)
    
    # Select features
    if feature_list is not None:
        available_features = [f for f in feature_list if f in train_processed.columns]
        print(f"Using {len(available_features)} selected features (out of {len(feature_list)} requested)")
        
        train_features = train_processed[available_features]
        test_features = test_processed[available_features]
    else:
        # Use all features except Id and SalePrice
        feature_cols = [col for col in train_processed.columns if col not in ['Id', 'SalePrice']]
        train_features = train_processed[feature_cols]
        test_features = test_processed[feature_cols]
        print(f"Using all {len(feature_cols)} features")
    
    return train_features, test_features, encoders

def run_lightgbm_cv(X, y, params=None):
    """Run LightGBM with cross-validation"""
    
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    print(f"\nLightGBM Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    models = []
    
    print(f"\nRunning 5-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predict and calculate RMSE
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        cv_scores.append(rmse)
        models.append(model)
        
        print(f"  Fold {fold}: RMSE = {rmse:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\nCV Results:")
    print(f"  Mean RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  R² equivalent: ~{1 - (cv_mean**2 / np.var(y)):.3f}")
    
    return models, cv_scores

def analyze_feature_importance(models, feature_names):
    """Analyze feature importance across CV folds"""
    print(f"\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Average importance across folds
    importance_sum = np.zeros(len(feature_names))
    
    for model in models:
        importance_sum += model.feature_importance()
    
    avg_importance = importance_sum / len(models)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    print(f"Top 20 Most Important Features:")
    print(f"{'Rank':<5} {'Feature':<20} {'Importance':<12}")
    print("-" * 40)
    
    for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
        print(f"{i:<5} {row['feature']:<20} {row['importance']:8.0f}")
    
    return importance_df

def compare_configurations(X, y):
    """Compare different LightGBM configurations"""
    print(f"\n" + "=" * 60)
    print("LIGHTGBM CONFIGURATION COMPARISON")
    print("=" * 60)
    
    configs = {
        'Conservative': {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': 42
        },
        'Moderate': {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        },
        'Aggressive': {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 63,
            'learning_rate': 0.08,
            'feature_fraction': 0.95,
            'bagging_fraction': 0.85,
            'bagging_freq': 3,
            'verbose': -1,
            'random_state': 42
        }
    }
    
    results = {}
    
    print(f"{'Configuration':<15} {'CV RMSE':<12} {'CV Std':<10}")
    print("-" * 40)
    
    for config_name, params in configs.items():
        models, cv_scores = run_lightgbm_cv(X, y, params)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        results[config_name] = {
            'models': models,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'params': params
        }
        
        print(f"{config_name:<15} {cv_mean:8.4f}     {cv_std:6.4f}")
    
    # Find best configuration
    best_config = min(results.keys(), key=lambda k: results[k]['cv_mean'])
    
    print(f"\nBest configuration: {best_config}")
    print(f"Best RMSE: {results[best_config]['cv_mean']:.4f} ± {results[best_config]['cv_std']:.4f}")
    
    return results, best_config

def create_submission(models, X_test, test_ids, filename):
    """Create submission using ensemble of CV models"""
    print(f"\n" + "=" * 40)
    print("CREATING SUBMISSION")
    print("=" * 40)
    
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
    print("HOUSE PRICES LIGHTGBM MODEL")
    print("=" * 60)
    
    # Load data
    train, test = load_data()
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Load selected features
    try:
        selected_features = load_selected_features()
        print(f"Loaded {len(selected_features)} selected features")
    except:
        print("Could not load selected features, using all features")
        selected_features = None
    
    # Prepare features
    X_train, X_test, encoders = prepare_features(train, test, selected_features)
    y_train = np.log1p(train['SalePrice'])  # Log transform target
    
    print(f"Final feature matrix: {X_train.shape}")
    print(f"Target: log(SalePrice), mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    
    # Compare different configurations
    results, best_config = compare_configurations(X_train, y_train)
    
    # Analyze feature importance for best model
    best_models = results[best_config]['models']
    importance_df = analyze_feature_importance(best_models, X_train.columns)
    
    # Create submission with best model
    test_ids = test['Id'].values
    submission = create_submission(best_models, X_test, test_ids, 'lightgbm_selected_features.csv')
    
    # Also compare with all features
    print(f"\n" + "=" * 60)
    print("COMPARISON: ALL FEATURES VS SELECTED FEATURES")
    print("=" * 60)
    
    # Run with all features
    X_train_all, X_test_all, _ = prepare_features(train, test, feature_list=None)
    print(f"All features model:")
    all_models, all_cv_scores = run_lightgbm_cv(X_train_all, y_train, results[best_config]['params'])
    
    print(f"\nPerformance Comparison:")
    print(f"Selected features ({len(X_train.columns)}): {results[best_config]['cv_mean']:.4f} ± {results[best_config]['cv_std']:.4f}")
    print(f"All features ({len(X_train_all.columns)}):     {np.mean(all_cv_scores):.4f} ± {np.std(all_cv_scores):.4f}")
    
    # Create submission with all features too
    submission_all = create_submission(all_models, X_test_all, test_ids, 'lightgbm_all_features.csv')
    
    # Save results
    importance_df.to_csv('lightgbm_feature_importance.csv', index=False)
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Files created:")
    print("  submissions/lightgbm_selected_features.csv")
    print("  submissions/lightgbm_all_features.csv")
    print("  lightgbm_feature_importance.csv")

if __name__ == "__main__":
    main()