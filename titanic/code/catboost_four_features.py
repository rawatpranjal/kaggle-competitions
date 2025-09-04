#!/usr/bin/env python3
"""
CATBOOST ON FOUR CORE FEATURES
==============================
Apply CatBoost to the winning 4-feature set: Sex_Male, Age_Child, CabinCount, LargeFamily
Test if CatBoost can improve on the 78.23% leaderboard score from LightGBM
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
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
    print("CATBOOST ON FOUR CORE FEATURES")
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
    
    # ========== BASELINE COMPARISON ==========
    print("\n" + "="*40)
    print("BASELINE COMPARISON")
    print("="*40)
    
    # Logistic Regression baseline
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_scores = cross_val_score(lr, X_train, y, cv=cv, scoring='accuracy')
    lr_mean = np.mean(lr_scores)
    
    print(f"Logistic Regression CV: {lr_mean:.3f} ± {np.std(lr_scores):.3f}")
    
    # ========== CATBOOST CONFIGURATIONS ==========
    print("\n" + "="*40)
    print("CATBOOST CONFIGURATIONS")
    print("="*40)
    
    # Test different CatBoost configurations
    catboost_configs = {
        'Conservative': {
            'iterations': 1000,
            'learning_rate': 0.01,
            'depth': 3,
            'l2_leaf_reg': 10.0,
            'border_count': 32,
            'random_strength': 1.0,
            'bagging_temperature': 0.0,
            'verbose': False,
            'random_seed': 42,
            'early_stopping_rounds': 100,
        },
        'Moderate': {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 4,
            'l2_leaf_reg': 5.0,
            'border_count': 64,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'verbose': False,
            'random_seed': 42,
            'early_stopping_rounds': 100,
        },
        'Aggressive': {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 1.0,
            'border_count': 128,
            'random_strength': 2.0,
            'bagging_temperature': 2.0,
            'verbose': False,
            'random_seed': 42,
            'early_stopping_rounds': 100,
        },
        'Default': {
            'iterations': 1000,
            'verbose': False,
            'random_seed': 42,
            'early_stopping_rounds': 100,
        }
    }
    
    best_config = None
    best_score = 0
    results = []
    
    for config_name, params in catboost_configs.items():
        print(f"\nTesting {config_name} configuration...")
        
        cb_scores = []
        feature_importance = np.zeros(len(X_train.columns))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Create CatBoost model
            model = CatBoostClassifier(**params)
            
            # Train model
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            
            # Predict and score
            val_pred = model.predict(X_val)
            fold_accuracy = (val_pred == y_val).mean()
            cb_scores.append(fold_accuracy)
            
            # Accumulate feature importance
            feature_importance += model.feature_importances_
            
            print(f"  Fold {fold+1}: {fold_accuracy:.4f} ({model.tree_count_} trees)")
        
        config_mean = np.mean(cb_scores)
        config_std = np.std(cb_scores)
        
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
    
    print(f"{'Model':<25} {'CV Accuracy':<12} {'Std Dev'}")
    print("-" * 50)
    print(f"{'Logistic Regression':<25} {lr_mean:8.3f}     {np.std(lr_scores):6.3f}")
    
    for result in sorted(results, key=lambda x: x['mean_accuracy'], reverse=True):
        is_best = "⭐" if result['config'] == best_config else ""
        print(f"{result['config'] + ' CatBoost':<25} {result['mean_accuracy']:8.3f}     {result['std_accuracy']:6.3f} {is_best}")
    
    # Best model analysis
    best_result = [r for r in results if r['config'] == best_config][0]
    improvement_vs_lr = best_result['mean_accuracy'] - lr_mean
    
    print(f"\nBest model: {best_config} CatBoost")
    print(f"Improvement over logistic regression: {improvement_vs_lr:+.3f}")
    
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
    best_params = catboost_configs[best_config]
    final_model = CatBoostClassifier(**best_params)
    
    # Train without validation set for final model
    final_params = best_params.copy()
    final_params.pop('early_stopping_rounds', None)  # Remove early stopping for final training
    final_model = CatBoostClassifier(**final_params)
    
    final_model.fit(X_train, y, verbose=False)
    
    print(f"Final model trained with {final_model.tree_count_} trees")
    
    # Make predictions
    test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = final_model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_pred
    })
    
    submission.to_csv('submission_catboost_four_features.csv', index=False)
    
    # ========== SUBMISSION ANALYSIS ==========
    print("\n" + "="*50)
    print("SUBMISSION ANALYSIS")
    print("="*50)
    
    print(f"• Model: {best_config} CatBoost on 4 core features")
    print(f"• CV Accuracy: {best_score:.3f}")
    print(f"• Features: {list(X_train.columns)}")
    print(f"• Trees in final model: {final_model.tree_count_}")
    print(f"• Test predictions: {len(test_pred)} samples")
    print(f"• Predicted survival rate: {test_pred.mean():.3f}")
    
    # Confidence analysis
    high_confidence = (test_pred_proba > 0.8) | (test_pred_proba < 0.2)
    print(f"• High confidence predictions: {high_confidence.sum()} ({high_confidence.mean():.1%})")
    
    # Compare with known results
    print(f"\nComparison with previous models:")
    print(f"• 4-feature Logistic (CV): {lr_mean:.3f}")
    print(f"• 4-feature Interaction (LB): 77.99%")
    print(f"• 4-feature LightGBM (LB): 78.23%")
    print(f"• 160-feature LightGBM (LB): 75.60%")
    print(f"• 4-feature CatBoost (CV): {best_score:.3f}")
    
    # Prediction breakdown
    print(f"\nPrediction breakdown:")
    print(f"  Survived: {test_pred.sum()} ({test_pred.mean():.1%})")
    print(f"  Died: {(test_pred == 0).sum()} ({(test_pred == 0).mean():.1%})")
    print(f"  Mean probability: {test_pred_proba.mean():.3f}")
    print(f"  Probability std: {test_pred_proba.std():.3f}")
    
    print(f"\nSubmission file: submission_catboost_four_features.csv")
    
    # Performance expectation
    if best_score > 0.825:
        print(f"Expected performance: Likely >78.5% (may set new best)")
    elif best_score > 0.820:
        print(f"Expected performance: Competitive with 78.23% (LightGBM)")
    else:
        print(f"Expected performance: Good but likely <78.2%")
    
    # ========== CATBOOST vs OTHER ALGORITHMS ==========
    print("\n" + "="*40)
    print("CATBOOST ADVANTAGES")
    print("="*40)
    
    print("CatBoost strengths:")
    print("• Excellent default parameters")
    print("• Handles categorical features natively")
    print("• Robust to overfitting")
    print("• Symmetric trees (good for small datasets)")
    print("• Built-in feature interactions")
    
    print(f"\nAlgorithm comparison on 4 features:")
    print(f"• Logistic Regression: {lr_mean:.3f} (interpretable)")
    print(f"• CatBoost: {best_score:.3f} (robust)")
    print(f"• LightGBM: 0.828 (fast, from previous run)")
    
    if best_score > lr_mean + 0.01:
        print(f"\n✅ CatBoost shows strong improvement (+{improvement_vs_lr:.1%})")
        print("Gradient boosting algorithms effective on this feature set")
    else:
        print(f"\n📊 CatBoost performs similarly to other methods")
        print("Feature set may be near optimal complexity")

    # Model interpretation
    print(f"\n" + "="*40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*40)
    
    # Normalize importance to percentages
    total_importance = importance_values.sum()
    importance_pct = (importance_values / total_importance) * 100
    
    print("Feature contribution to model:")
    for feature, pct in zip(feature_names, importance_pct):
        print(f"  {feature:15}: {pct:5.1f}%")
    
    # Rank features by importance
    feature_ranking = sorted(zip(feature_names, importance_values), key=lambda x: x[1], reverse=True)
    print(f"\nFeature ranking:")
    for i, (feature, importance) in enumerate(feature_ranking, 1):
        print(f"  {i}. {feature}")

if __name__ == "__main__":
    main()