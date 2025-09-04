#!/usr/bin/env python3
"""
LIGHTGBM ON FOUR CORE FEATURES
==============================
Apply LightGBM to the winning 4-feature set: Sex_Male, Age_Child, CabinCount, LargeFamily
Test if gradient boosting can improve on the 77.99% leaderboard score
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train and test data"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def create_four_features(df):
    """Create the winning four features"""
    features = {}
    
    # Core features
    age_filled = df['Age'].fillna(df['Age'].median())
    features['Sex_Male'] = (df['Sex'] == 'male').astype(int)
    features['Age_Child'] = (age_filled < 12).astype(int)
    
    # Cabin count
    cabin_values = df['Cabin'].fillna('Unknown')
    cabin_count = cabin_values.str.count(' ') + 1
    cabin_count[df['Cabin'].isna()] = 0
    features['CabinCount'] = cabin_count
    
    # Large family indicator
    family_size = df['SibSp'] + df['Parch'] + 1
    features['LargeFamily'] = (family_size > 4).astype(int)
    
    return features

def create_features_for_both(train_df, test_df):
    """Create features for both train and test, ensuring consistency"""
    
    # Combine for consistent feature engineering
    all_data = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # Create features
    features_dict = create_four_features(all_data)
    feature_df = pd.DataFrame(features_dict)
    
    # Split back
    train_features = feature_df.iloc[:len(train_df)].reset_index(drop=True)
    test_features = feature_df.iloc[len(train_df):].reset_index(drop=True)
    
    return train_features, test_features

def main():
    print("="*60)
    print("LIGHTGBM ON FOUR CORE FEATURES")
    print("="*60)
    
    # Load data
    train, test = load_data()
    y = train['Survived'].values
    
    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")
    print(f"Survival rate: {y.mean():.3f}")
    
    # Create features
    X_train, X_test = create_features_for_both(train, test)
    
    print(f"\nFeature set: {list(X_train.columns)}")
    print(f"Total features: {len(X_train.columns)}")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ========== BASELINE LOGISTIC REGRESSION ==========
    print("\n" + "="*40)
    print("BASELINE: LOGISTIC REGRESSION")
    print("="*40)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_scores = cross_val_score(lr, X_train, y, cv=cv, scoring='accuracy')
    lr_mean = np.mean(lr_scores)
    
    print(f"Logistic Regression CV: {lr_mean:.3f} ± {np.std(lr_scores):.3f}")
    
    # ========== LIGHTGBM CONFIGURATIONS ==========
    print("\n" + "="*40)
    print("LIGHTGBM CONFIGURATIONS")
    print("="*40)
    
    # Test different LightGBM configurations
    lgb_configs = {
        'Conservative': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 4,      # Very simple trees
            'learning_rate': 0.01,
            'feature_fraction': 1.0,  # Use all 4 features
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'min_data_in_leaf': 50,  # Conservative
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'max_depth': 3,
        },
        'Moderate': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 8,
            'learning_rate': 0.05,
            'feature_fraction': 1.0,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'min_data_in_leaf': 30,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_depth': 4,
        },
        'Aggressive': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'feature_fraction': 1.0,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'min_data_in_leaf': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': 5,
        }
    }
    
    best_config = None
    best_score = 0
    results = []
    
    for config_name, params in lgb_configs.items():
        print(f"\nTesting {config_name} configuration...")
        
        lgb_scores = []
        feature_importance = np.zeros(len(X_train.columns))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Create datasets
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            # Predict and score
            val_pred = model.predict(X_val)
            val_binary = (val_pred > 0.5).astype(int)
            fold_accuracy = (val_binary == y_val).mean()
            lgb_scores.append(fold_accuracy)
            
            # Accumulate feature importance
            feature_importance += model.feature_importance(importance_type='gain')
            
            print(f"  Fold {fold+1}: {fold_accuracy:.4f}")
        
        config_mean = np.mean(lgb_scores)
        config_std = np.std(lgb_scores)
        
        print(f"  {config_name} CV: {config_mean:.3f} ± {config_std:.3f}")
        
        results.append({
            'config': config_name,
            'mean_accuracy': config_mean,
            'std_accuracy': config_std,
            'feature_importance': feature_importance / 5
        })
        
        if config_mean > best_score:
            best_score = config_mean
            best_config = config_name
    
    # ========== RESULTS COMPARISON ==========
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    print(f"{'Model':<20} {'CV Accuracy':<12} {'Std Dev'}")
    print("-" * 45)
    print(f"{'Logistic Regression':<20} {lr_mean:8.3f}     {np.std(lr_scores):6.3f}")
    
    for result in sorted(results, key=lambda x: x['mean_accuracy'], reverse=True):
        is_best = "⭐" if result['config'] == best_config else ""
        print(f"{result['config'] + ' LightGBM':<20} {result['mean_accuracy']:8.3f}     {result['std_accuracy']:6.3f} {is_best}")
    
    # Best model analysis
    best_result = [r for r in results if r['config'] == best_config][0]
    improvement = best_result['mean_accuracy'] - lr_mean
    
    print(f"\nBest model: {best_config} LightGBM")
    print(f"Improvement over logistic regression: {improvement:+.3f}")
    
    # Feature importance
    print(f"\nFeature importance (best model):")
    feature_names = X_train.columns
    importance_values = best_result['feature_importance']
    
    for i, (feature, importance) in enumerate(zip(feature_names, importance_values)):
        print(f"  {i+1}. {feature:15}: {importance:8.1f}")
    
    # ========== FINAL MODEL TRAINING ==========
    print("\n" + "="*40)
    print("TRAINING FINAL MODEL")
    print("="*40)
    
    # Train best configuration on full training set
    best_params = lgb_configs[best_config]
    train_data = lgb.Dataset(X_train, label=y)
    
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(100)]
    )
    
    print(f"Final model trained with {final_model.num_trees()} trees")
    
    # Make predictions
    test_pred = final_model.predict(X_test)
    test_binary = (test_pred > 0.5).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_binary
    })
    
    submission.to_csv('submission_lightgbm_four_features.csv', index=False)
    
    # ========== SUBMISSION ANALYSIS ==========
    print("\n" + "="*50)
    print("SUBMISSION ANALYSIS")
    print("="*50)
    
    print(f"• Model: {best_config} LightGBM on 4 core features")
    print(f"• CV Accuracy: {best_score:.3f}")
    print(f"• Features: {list(X_train.columns)}")
    print(f"• Trees in final model: {final_model.num_trees()}")
    print(f"• Test predictions: {len(test_binary)} samples")
    print(f"• Predicted survival rate: {test_binary.mean():.3f}")
    
    # Confidence analysis
    high_confidence = (test_pred > 0.8) | (test_pred < 0.2)
    print(f"• High confidence predictions: {high_confidence.sum()} ({high_confidence.mean():.1%})")
    
    # Compare with known results
    print(f"\nComparison with previous models:")
    print(f"• 4-feature logistic (CV): {lr_mean:.3f}")
    print(f"• 4-feature interaction (LB): 77.99%")
    print(f"• 160-feature LightGBM (LB): 75.60%")
    print(f"• 4-feature LightGBM (CV): {best_score:.3f}")
    
    # Prediction breakdown
    print(f"\nPrediction breakdown:")
    print(f"  Survived: {test_binary.sum()} ({test_binary.mean():.1%})")
    print(f"  Died: {(test_binary == 0).sum()} ({(test_binary == 0).mean():.1%})")
    
    print(f"\nSubmission file: submission_lightgbm_four_features.csv")
    print(f"Expected performance: Better than 75.60% (160 features), competitive with 77.99% (interaction model)")
    
    # ========== FEATURE INTERACTION ANALYSIS ==========
    print("\n" + "="*40)
    print("LIGHTGBM vs LOGISTIC REGRESSION")
    print("="*40)
    
    print("Advantages of LightGBM:")
    print("• Automatic feature interactions")
    print("• Non-linear relationships")
    print("• Handles feature scaling automatically")
    print("• Robust to outliers")
    
    print("\nAdvantages of Logistic Regression:")
    print("• Interpretable coefficients")
    print("• Explicit interaction terms")
    print("• Less prone to overfitting")
    print("• Probabilistic interpretation")
    
    if improvement > 0.01:
        print(f"\n✅ LightGBM shows meaningful improvement (+{improvement:.1%})")
        print("Gradient boosting found patterns logistic regression missed")
    else:
        print(f"\n📊 Models perform similarly (diff: {improvement:+.1%})")
        print("Feature set may be optimal for linear relationships")

if __name__ == "__main__":
    main()