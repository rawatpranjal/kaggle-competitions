#!/usr/bin/env python3
"""
L1 REGULARIZED LOGISTIC REGRESSION
==================================
Apply L1 (Lasso) regularization to find minimal feature set
from the saturated model with all interactions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import itertools
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
    
    age_filled = df['Age'].fillna(df['Age'].median())
    features['Sex_Male'] = (df['Sex'] == 'male').astype(int)
    features['Age_Child'] = (age_filled < 12).astype(int)
    
    cabin_values = df['Cabin'].fillna('Unknown')
    cabin_count = cabin_values.str.count(' ') + 1
    cabin_count[df['Cabin'].isna()] = 0
    features['CabinCount'] = cabin_count
    
    family_size = df['SibSp'] + df['Parch'] + 1
    features['LargeFamily'] = (family_size > 4).astype(int)
    
    return features

def create_saturated_model(feature_df):
    """Create fully saturated model with ALL possible interactions"""
    base_features = ['Sex_Male', 'Age_Child', 'CabinCount', 'LargeFamily']
    saturated_df = feature_df[base_features].copy()
    
    # Two-way interactions
    for i, feat1 in enumerate(base_features):
        for feat2 in base_features[i+1:]:
            interaction_name = f"{feat1}_x_{feat2}"
            saturated_df[interaction_name] = feature_df[feat1] * feature_df[feat2]
    
    # Three-way interactions
    for combo in itertools.combinations(base_features, 3):
        interaction_name = f"{combo[0]}_x_{combo[1]}_x_{combo[2]}"
        saturated_df[interaction_name] = (feature_df[combo[0]] * 
                                         feature_df[combo[1]] * 
                                         feature_df[combo[2]])
    
    # Four-way interaction
    four_way_name = "Sex_Male_x_Age_Child_x_CabinCount_x_LargeFamily"
    saturated_df[four_way_name] = (feature_df['Sex_Male'] * 
                                   feature_df['Age_Child'] * 
                                   feature_df['CabinCount'] * 
                                   feature_df['LargeFamily'])
    
    return saturated_df

def create_features_for_both(train_df, test_df):
    """Create features for both train and test, ensuring consistency"""
    all_data = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    features_dict = create_four_features(all_data)
    feature_df = pd.DataFrame(features_dict)
    saturated_df = create_saturated_model(feature_df)
    
    train_features = saturated_df.iloc[:len(train_df)].reset_index(drop=True)
    test_features = saturated_df.iloc[len(train_df):].reset_index(drop=True)
    
    return train_features, test_features

def l1_regularization_path(X, y, cv):
    """Test different L1 regularization strengths"""
    
    print("="*60)
    print("L1 REGULARIZATION PATH ANALYSIS")
    print("="*60)
    
    # Range of C values (inverse of regularization strength)
    C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    
    results = []
    
    print(f"{'C Value':<10} {'Active Features':<15} {'CV Accuracy':<12} {'CV Std':<8} {'Selected Features'}")
    print("-" * 90)
    
    for C in C_values:
        # L1 logistic regression
        lr = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42, max_iter=1000)
        
        # Cross-validation
        cv_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Fit on full data to see selected features
        lr.fit(X, y)
        active_features = np.sum(np.abs(lr.coef_[0]) > 1e-6)
        
        # Get selected feature names
        selected_mask = np.abs(lr.coef_[0]) > 1e-6
        selected_features = X.columns[selected_mask].tolist()
        
        results.append({
            'C': C,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'active_features': active_features,
            'selected_features': selected_features,
            'coefficients': lr.coef_[0][selected_mask],
            'intercept': lr.intercept_[0]
        })
        
        # Truncate feature list for display
        feature_display = selected_features[:3]
        if len(selected_features) > 3:
            feature_display.append(f"... +{len(selected_features)-3} more")
        
        print(f"{C:<10.3f} {active_features:<15d} {cv_mean:<12.3f} {cv_std:<8.3f} {', '.join(feature_display)}")
    
    return results

def analyze_optimal_models(results):
    """Analyze the best performing models"""
    
    print("\n" + "="*60)
    print("OPTIMAL MODEL ANALYSIS")
    print("="*60)
    
    # Find best CV score
    best_idx = np.argmax([r['cv_mean'] for r in results])
    best_result = results[best_idx]
    
    print(f"Best CV Performance:")
    print(f"  C = {best_result['C']}")
    print(f"  CV Accuracy: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
    print(f"  Active Features: {best_result['active_features']}")
    
    # Find most parsimonious model with good performance
    # (within 1 std of best)
    threshold = best_result['cv_mean'] - best_result['cv_std']
    good_models = [r for r in results if r['cv_mean'] >= threshold]
    most_parsimonious = min(good_models, key=lambda x: x['active_features'])
    
    print(f"\nMost Parsimonious Good Model:")
    print(f"  C = {most_parsimonious['C']}")
    print(f"  CV Accuracy: {most_parsimonious['cv_mean']:.3f} ± {most_parsimonious['cv_std']:.3f}")
    print(f"  Active Features: {most_parsimonious['active_features']}")
    
    # Show detailed coefficients for both models
    for title, result in [("Best Performance Model", best_result), 
                         ("Most Parsimonious Model", most_parsimonious)]:
        print(f"\n{title} Coefficients:")
        print(f"{'Feature':<40} {'Coefficient':<12} {'Odds Ratio':<12}")
        print("-" * 65)
        
        for feat, coef in zip(result['selected_features'], result['coefficients']):
            odds_ratio = np.exp(coef)
            print(f"{feat:<40} {coef:8.3f}     {odds_ratio:8.2f}")
        print(f"Intercept: {result['intercept']:.3f}")
    
    return best_result, most_parsimonious

def create_submission(X_train, X_test, y, result, model_name):
    """Create submission with selected model"""
    
    # Train final model
    lr = LogisticRegression(penalty='l1', C=result['C'], solver='liblinear', 
                           random_state=42, max_iter=1000)
    lr.fit(X_train, y)
    
    # Select features that were kept
    selected_mask = np.abs(lr.coef_[0]) > 1e-6
    X_train_selected = X_train.iloc[:, selected_mask]
    X_test_selected = X_test.iloc[:, selected_mask]
    
    # Retrain on selected features only
    lr_final = LogisticRegression(penalty='l1', C=result['C'], solver='liblinear',
                                 random_state=42, max_iter=1000)
    lr_final.fit(X_train_selected, y)
    
    # Make predictions
    test_pred_proba = lr_final.predict_proba(X_test_selected)[:, 1]
    test_pred = (test_pred_proba > 0.5).astype(int)
    
    return test_pred, test_pred_proba, selected_mask

def main():
    print("="*60)
    print("L1 REGULARIZED LOGISTIC REGRESSION")
    print("="*60)
    
    # Load data
    train, test = load_data()
    y = train['Survived'].values
    
    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")
    print(f"Survival rate: {y.mean():.3f}")
    
    # Create features
    X_train, X_test = create_features_for_both(train, test)
    
    print(f"\nSaturated model shape: {X_train.shape}")
    print(f"Total features: {len(X_train.columns)}")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # L1 regularization path
    results = l1_regularization_path(X_train, y, cv)
    
    # Analyze optimal models
    best_result, parsimonious_result = analyze_optimal_models(results)
    
    # Create submissions for both models
    print("\n" + "="*50)
    print("CREATING SUBMISSIONS")
    print("="*50)
    
    # Best performance model
    best_pred, best_proba, best_mask = create_submission(
        X_train, X_test, y, best_result, "best"
    )
    
    best_submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': best_pred
    })
    best_submission.to_csv('submission_l1_best.csv', index=False)
    
    print(f"Best Model Submission:")
    print(f"  Features: {best_result['active_features']}")
    print(f"  CV Accuracy: {best_result['cv_mean']:.3f}")
    print(f"  Predicted survival rate: {best_pred.mean():.3f}")
    print(f"  File: submission_l1_best.csv")
    
    # Parsimonious model (if different)
    if parsimonious_result['C'] != best_result['C']:
        pars_pred, pars_proba, pars_mask = create_submission(
            X_train, X_test, y, parsimonious_result, "parsimonious"
        )
        
        pars_submission = pd.DataFrame({
            'PassengerId': test['PassengerId'],
            'Survived': pars_pred
        })
        pars_submission.to_csv('submission_l1_parsimonious.csv', index=False)
        
        print(f"\nParsimonious Model Submission:")
        print(f"  Features: {parsimonious_result['active_features']}")
        print(f"  CV Accuracy: {parsimonious_result['cv_mean']:.3f}")
        print(f"  Predicted survival rate: {pars_pred.mean():.3f}")
        print(f"  File: submission_l1_parsimonious.csv")
        
        # Compare predictions
        diff_count = (best_pred != pars_pred).sum()
        print(f"  Prediction differences: {diff_count} ({diff_count/len(best_pred):.1%})")
    else:
        print("\nBest and parsimonious models are identical.")
    
    # Feature selection summary
    print("\n" + "="*50)
    print("FEATURE SELECTION SUMMARY")
    print("="*50)
    
    print("Regularization insights:")
    print(f"• No regularization (C=∞): 15 features, 82.5% CV")
    print(f"• Optimal regularization: {best_result['active_features']} features, {best_result['cv_mean']:.1%} CV")
    print(f"• Strong regularization: {results[0]['active_features']} features, {results[0]['cv_mean']:.1%} CV")
    
    # Show regularization path summary
    feature_counts = [r['active_features'] for r in results]
    accuracies = [r['cv_mean'] for r in results]
    
    print(f"\nRegularization path:")
    for i, (count, acc) in enumerate(zip(feature_counts, accuracies)):
        if i == 0 or count != feature_counts[i-1]:
            print(f"  {count} features: {acc:.1%} CV")

if __name__ == "__main__":
    main()