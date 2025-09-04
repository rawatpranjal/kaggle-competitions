#!/usr/bin/env python3
"""
BOX-COX TRANSFORMATION TEST
===========================
Test Box-Cox transformation for optimal target variable normalization.
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

def find_optimal_box_cox_lambda(y):
    """Find optimal Box-Cox lambda parameter"""
    print("=" * 50)
    print("BOX-COX LAMBDA OPTIMIZATION")
    print("=" * 50)
    
    # Ensure all values are positive (Box-Cox requirement)
    y_positive = y + 1  # Add 1 to ensure positivity
    
    # Find optimal lambda
    transformed_data, fitted_lambda = stats.boxcox(y_positive)
    
    print(f"Original target statistics:")
    print(f"  Min: ${y.min():,.0f}")
    print(f"  Max: ${y.max():,.0f}")
    print(f"  Mean: ${y.mean():,.0f}")
    print(f"  Skewness: {y.skew():.3f}")
    print(f"  Kurtosis: {y.kurtosis():.3f}")
    
    print(f"\nOptimal Box-Cox lambda: {fitted_lambda:.4f}")
    print(f"Interpretation:")
    if abs(fitted_lambda) < 0.01:
        print(f"  λ ≈ 0: Suggests log transformation")
    elif abs(fitted_lambda - 0.5) < 0.1:
        print(f"  λ ≈ 0.5: Suggests square root transformation")
    elif abs(fitted_lambda - 1) < 0.1:
        print(f"  λ ≈ 1: Suggests no transformation needed")
    else:
        print(f"  λ = {fitted_lambda:.3f}: Custom power transformation")
    
    print(f"\nBox-Cox transformed statistics:")
    print(f"  Mean: {transformed_data.mean():.3f}")
    print(f"  Std: {transformed_data.std():.3f}")
    print(f"  Skewness: {stats.skew(transformed_data):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(transformed_data):.3f}")
    
    # Compare with log transformation
    log_y = np.log1p(y)
    print(f"\nComparison with log(1+y):")
    print(f"  Log skewness: {log_y.skew():.3f}")
    print(f"  Box-Cox skewness: {stats.skew(transformed_data):.3f}")
    print(f"  Box-Cox is better: {abs(stats.skew(transformed_data)) < abs(log_y.skew())}")
    
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

def cv_with_box_cox(train, test, lam):
    """Cross-validation with Box-Cox transformation"""
    print(f"\n" + "=" * 50)
    print(f"CATBOOST WITH BOX-COX (λ={lam:.4f})")
    print("=" * 50)
    
    X_train, X_test, cat_features = prepare_features_for_catboost(train, test)
    y_train_original = train['SalePrice']
    y_train_transformed = box_cox_transform(y_train_original, lam)
    
    print(f"Box-Cox transformed target statistics:")
    print(f"  Mean: {y_train_transformed.mean():.3f}")
    print(f"  Std: {y_train_transformed.std():.3f}")
    print(f"  Skewness: {stats.skew(y_train_transformed):.3f}")
    
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
    models = []
    
    print(f"\nCross-validation:")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train_transformed.iloc[train_idx]
        y_val_transformed = y_train_transformed.iloc[val_idx]
        y_val_original = y_train_original.iloc[val_idx]
        
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val_transformed), verbose=False)
        
        # Predict in transformed space
        pred_transformed = model.predict(X_val)
        
        # Calculate RMSE in transformed space
        rmse_transformed = np.sqrt(mean_squared_error(y_val_transformed, pred_transformed))
        cv_scores_transformed.append(rmse_transformed)
        
        # Transform back to original space
        pred_original = inverse_box_cox_transform(pred_transformed, lam)
        pred_original = np.maximum(pred_original, 1000)  # Ensure positive prices
        
        # Calculate RMSE in original space
        rmse_original = np.sqrt(mean_squared_error(y_val_original, pred_original))
        cv_scores_original.append(rmse_original)
        
        models.append(model)
        
        print(f"  Fold {fold}: Transformed RMSE = {rmse_transformed:.4f}, Original RMSE = ${rmse_original:,.0f}")
    
    cv_mean_transformed = np.mean(cv_scores_transformed)
    cv_std_transformed = np.std(cv_scores_transformed)
    cv_mean_original = np.mean(cv_scores_original)
    cv_std_original = np.std(cv_scores_original)
    
    print(f"\nCV Results:")
    print(f"  Transformed space: {cv_mean_transformed:.4f} ± {cv_std_transformed:.4f}")
    print(f"  Original space: ${cv_mean_original:,.0f} ± ${cv_std_original:,.0f}")
    
    return models, cv_mean_transformed, cv_std_transformed, cv_mean_original, cv_std_original

def compare_transformations(train, test):
    """Compare different transformations"""
    print(f"\n" + "=" * 60)
    print("TRANSFORMATION COMPARISON")
    print("=" * 60)
    
    # Get optimal Box-Cox lambda
    y = train['SalePrice']
    optimal_lambda, _ = find_optimal_box_cox_lambda(y)
    
    # Test different transformations
    transformations = [
        ('box_cox_optimal', optimal_lambda, f"Box-Cox (λ={optimal_lambda:.3f})"),
        ('box_cox_sqrt', 0.5, "Box-Cox (λ=0.5, sqrt-like)"),
        ('box_cox_log', 0.0, "Box-Cox (λ=0.0, log-like)"),
        ('log1p', None, "Log(1+y)")
    ]
    
    results = {}
    
    for transform_name, lam, description in transformations:
        print(f"\nTesting {description}:")
        
        if transform_name == 'log1p':
            # Use our previous log1p results
            X_train, X_test, cat_features = prepare_features_for_catboost(train, test)
            y_train_log = np.log1p(train['SalePrice'])
            
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
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                y_val_orig = y.iloc[val_idx]
                
                model = CatBoostRegressor(cat_features=cat_features, **params)
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val_log), verbose=False)
                
                pred_log = model.predict(X_val)
                rmse_log = np.sqrt(mean_squared_error(y_val_log, pred_log))
                cv_scores.append(rmse_log)
                
                pred_orig = np.expm1(pred_log)
                rmse_orig = np.sqrt(mean_squared_error(y_val_orig, pred_orig))
                cv_scores_original.append(rmse_orig)
            
            cv_mean = np.mean(cv_scores)
            cv_mean_orig = np.mean(cv_scores_original)
            print(f"  Log space RMSE: {cv_mean:.4f}")
            print(f"  Original space RMSE: ${cv_mean_orig:,.0f}")
            
            results[transform_name] = {
                'transformed_rmse': cv_mean,
                'original_rmse': cv_mean_orig,
                'description': description
            }
        else:
            _, cv_mean_transformed, _, cv_mean_original, _ = cv_with_box_cox(train, test, lam)
            
            results[transform_name] = {
                'transformed_rmse': cv_mean_transformed,
                'original_rmse': cv_mean_original,
                'description': description
            }
    
    return results, optimal_lambda

def create_box_cox_submission(train, test, lam):
    """Create submission with Box-Cox transformation"""
    print(f"\n" + "=" * 50)
    print(f"CREATING BOX-COX SUBMISSION (λ={lam:.4f})")
    print("=" * 50)
    
    X_train, X_test, cat_features = prepare_features_for_catboost(train, test)
    y_train_original = train['SalePrice']
    y_train_transformed = box_cox_transform(y_train_original, lam)
    
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
    model.fit(X_train, y_train_transformed)
    
    # Predict in transformed space
    test_pred_transformed = model.predict(X_test)
    
    # Transform back to original space
    test_pred_original = inverse_box_cox_transform(test_pred_transformed, lam)
    test_pred_original = np.maximum(test_pred_original, 1000)  # Ensure positive
    
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': test_pred_original
    })
    
    submission.to_csv(f'submissions/catboost_boxcox_{lam:.3f}.csv', index=False)
    
    print(f"Submission created: catboost_boxcox_{lam:.3f}.csv")
    print(f"Price range: ${test_pred_original.min():,.0f} - ${test_pred_original.max():,.0f}")
    print(f"Price median: ${np.median(test_pred_original):,.0f}")
    
    return submission

def main():
    print("=" * 60)
    print("BOX-COX TRANSFORMATION TEST")
    print("=" * 60)
    
    train, test = load_data()
    print(f"Dataset: {train.shape[0]} train, {test.shape[0]} test")
    
    # Compare transformations
    results, optimal_lambda = compare_transformations(train, test)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("TRANSFORMATION COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"{'Transformation':<25} {'Transformed RMSE':<17} {'Original RMSE':<15}")
    print("-" * 60)
    
    for transform_name, result in results.items():
        transformed_rmse = result['transformed_rmse']
        original_rmse = result['original_rmse']
        description = result['description']
        
        print(f"{description:<25} {transformed_rmse:11.4f}      ${original_rmse:8,.0f}")
    
    # Find best transformation
    best_transform = min(results.keys(), key=lambda k: results[k]['transformed_rmse'])
    best_result = results[best_transform]
    
    print(f"\nBest transformation: {best_result['description']}")
    print(f"Best RMSE: {best_result['transformed_rmse']:.4f}")
    
    # Create submission with best transformation
    if best_transform.startswith('box_cox') and best_transform != 'box_cox_log':
        if best_transform == 'box_cox_optimal':
            lam = optimal_lambda
        elif best_transform == 'box_cox_sqrt':
            lam = 0.5
        else:
            lam = 0.0
            
        submission = create_box_cox_submission(train, test, lam)
    else:
        print("Log transformation is still best - no new submission needed")
    
    print(f"\n" + "=" * 50)
    print("BOX-COX ANALYSIS COMPLETE")
    print("=" * 50)
    
    print("Key insights:")
    print("- Box-Cox finds optimal power transformation automatically")
    print(f"- Optimal λ = {optimal_lambda:.4f}")
    if abs(optimal_lambda) < 0.1:
        print("- Close to log transformation (λ ≈ 0)")
    elif abs(optimal_lambda - 0.5) < 0.1:
        print("- Close to square root transformation (λ ≈ 0.5)")
    else:
        print(f"- Custom power transformation with λ = {optimal_lambda:.3f}")

if __name__ == "__main__":
    main()