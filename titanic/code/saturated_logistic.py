#!/usr/bin/env python3
"""
SATURATED LOGISTIC REGRESSION MODEL
===================================
Create all possible interactions from 4 features:
- 4 main effects
- 6 two-way interactions (C(4,2) = 6)
- 4 three-way interactions (C(4,3) = 4)  
- 1 four-way interaction (C(4,4) = 1)
Total: 15 parameters (saturated model)
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

def create_saturated_model(feature_df):
    """Create fully saturated model with ALL possible interactions"""
    base_features = ['Sex_Male', 'Age_Child', 'CabinCount', 'LargeFamily']
    saturated_df = feature_df[base_features].copy()
    
    interaction_terms = []
    
    print("Creating interaction terms:")
    print("=" * 50)
    
    # Two-way interactions (6 terms)
    print("Two-way interactions:")
    two_way_terms = []
    for i, feat1 in enumerate(base_features):
        for feat2 in base_features[i+1:]:
            interaction_name = f"{feat1}_x_{feat2}"
            saturated_df[interaction_name] = feature_df[feat1] * feature_df[feat2]
            two_way_terms.append(interaction_name)
            print(f"  {interaction_name}")
    
    # Three-way interactions (4 terms)
    print("\nThree-way interactions:")
    three_way_terms = []
    for combo in itertools.combinations(base_features, 3):
        interaction_name = f"{combo[0]}_x_{combo[1]}_x_{combo[2]}"
        saturated_df[interaction_name] = (feature_df[combo[0]] * 
                                         feature_df[combo[1]] * 
                                         feature_df[combo[2]])
        three_way_terms.append(interaction_name)
        print(f"  {interaction_name}")
    
    # Four-way interaction (1 term)
    print("\nFour-way interaction:")
    four_way_name = "Sex_Male_x_Age_Child_x_CabinCount_x_LargeFamily"
    saturated_df[four_way_name] = (feature_df['Sex_Male'] * 
                                   feature_df['Age_Child'] * 
                                   feature_df['CabinCount'] * 
                                   feature_df['LargeFamily'])
    four_way_terms = [four_way_name]
    print(f"  {four_way_name}")
    
    # Summary
    print(f"\nModel structure:")
    print(f"  Main effects: {len(base_features)}")
    print(f"  Two-way interactions: {len(two_way_terms)}")
    print(f"  Three-way interactions: {len(three_way_terms)}")
    print(f"  Four-way interaction: {len(four_way_terms)}")
    print(f"  Total parameters: {len(saturated_df.columns)} + 1 (intercept) = {len(saturated_df.columns) + 1}")
    
    interaction_terms = {
        'main_effects': base_features,
        'two_way': two_way_terms,
        'three_way': three_way_terms,
        'four_way': four_way_terms
    }
    
    return saturated_df, interaction_terms

def create_features_for_both(train_df, test_df):
    """Create features for both train and test, ensuring consistency"""
    
    # Combine for consistent feature engineering
    all_data = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # Create base features
    features_dict = create_four_features(all_data)
    feature_df = pd.DataFrame(features_dict)
    
    # Create saturated model
    saturated_df, interaction_terms = create_saturated_model(feature_df)
    
    # Split back
    train_features = saturated_df.iloc[:len(train_df)].reset_index(drop=True)
    test_features = saturated_df.iloc[len(train_df):].reset_index(drop=True)
    
    return train_features, test_features, interaction_terms

def analyze_model_hierarchy(X, y, interaction_terms, cv):
    """Test model performance at each level of interaction complexity"""
    
    print("\n" + "="*60)
    print("HIERARCHICAL MODEL ANALYSIS")
    print("="*60)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    results = {}
    
    # Main effects only
    main_features = interaction_terms['main_effects']
    X_main = X[main_features]
    main_scores = cross_val_score(lr, X_main, y, cv=cv, scoring='accuracy')
    results['main_only'] = {
        'features': main_features,
        'n_params': len(main_features) + 1,
        'cv_mean': np.mean(main_scores),
        'cv_std': np.std(main_scores),
        'scores': main_scores
    }
    
    # Main + Two-way
    two_way_features = main_features + interaction_terms['two_way']
    X_two_way = X[two_way_features]
    two_way_scores = cross_val_score(lr, X_two_way, y, cv=cv, scoring='accuracy')
    results['main_plus_two_way'] = {
        'features': two_way_features,
        'n_params': len(two_way_features) + 1,
        'cv_mean': np.mean(two_way_scores),
        'cv_std': np.std(two_way_scores),
        'scores': two_way_scores
    }
    
    # Main + Two-way + Three-way
    three_way_features = two_way_features + interaction_terms['three_way']
    X_three_way = X[three_way_features]
    three_way_scores = cross_val_score(lr, X_three_way, y, cv=cv, scoring='accuracy')
    results['main_plus_two_plus_three_way'] = {
        'features': three_way_features,
        'n_params': len(three_way_features) + 1,
        'cv_mean': np.mean(three_way_scores),
        'cv_std': np.std(three_way_scores),
        'scores': three_way_scores
    }
    
    # Saturated (all interactions)
    saturated_features = list(X.columns)
    saturated_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    results['saturated'] = {
        'features': saturated_features,
        'n_params': len(saturated_features) + 1,
        'cv_mean': np.mean(saturated_scores),
        'cv_std': np.std(saturated_scores),
        'scores': saturated_scores
    }
    
    # Display results
    print(f"{'Model Level':<25} {'Parameters':<12} {'CV Accuracy':<15} {'CV Std':<10}")
    print("-" * 70)
    
    model_names = ['main_only', 'main_plus_two_way', 'main_plus_two_plus_three_way', 'saturated']
    model_labels = ['Main Effects Only', 'Main + Two-way', 'Main + Two + Three-way', 'Saturated (All)']
    
    for name, label in zip(model_names, model_labels):
        result = results[name]
        print(f"{label:<25} {result['n_params']:8d}     {result['cv_mean']:8.3f}      {result['cv_std']:7.3f}")
    
    # Find best model
    best_model = max(model_names, key=lambda x: results[x]['cv_mean'])
    best_result = results[best_model]
    
    print(f"\nBest performing model: {best_model}")
    print(f"Best CV accuracy: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
    
    return results, best_model

def main():
    print("="*60)
    print("SATURATED LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    # Load data
    train, test = load_data()
    y = train['Survived'].values
    
    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")
    print(f"Survival rate: {y.mean():.3f}")
    
    # Create features
    X_train, X_test, interaction_terms = create_features_for_both(train, test)
    
    print(f"\nSaturated model shape: {X_train.shape}")
    print(f"Total features: {len(X_train.columns)}")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Hierarchical analysis
    results, best_model = analyze_model_hierarchy(X_train, y, interaction_terms, cv)
    
    # ========== DETAILED COEFFICIENT ANALYSIS ==========
    print("\n" + "="*60)
    print("SATURATED MODEL COEFFICIENT ANALYSIS")
    print("="*60)
    
    # Fit saturated model for coefficient analysis
    lr_saturated = LogisticRegression(random_state=42, max_iter=1000)
    lr_saturated.fit(X_train, y)
    
    print(f"Saturated Model Coefficients:")
    print(f"{'Feature':<40} {'Coefficient':<12} {'Odds Ratio':<12} {'|Coef|'}")
    print("-" * 80)
    
    # Sort by absolute coefficient value
    coef_data = []
    for feature, coef in zip(X_train.columns, lr_saturated.coef_[0]):
        odds_ratio = np.exp(coef)
        abs_coef = abs(coef)
        coef_data.append((feature, coef, odds_ratio, abs_coef))
    
    # Sort by absolute coefficient value
    coef_data.sort(key=lambda x: x[3], reverse=True)
    
    for feature, coef, odds_ratio, abs_coef in coef_data:
        print(f"{feature:<40} {coef:8.3f}     {odds_ratio:8.2f}     {abs_coef:6.3f}")
    
    print(f"Intercept: {lr_saturated.intercept_[0]:.3f}")
    
    # ========== STATISTICAL SIGNIFICANCE ==========
    print("\n" + "="*50)
    print("INTERACTION SIGNIFICANCE ANALYSIS")
    print("="*50)
    
    # Compare model levels for significance
    main_acc = results['main_only']['cv_mean']
    two_way_acc = results['main_plus_two_way']['cv_mean']
    three_way_acc = results['main_plus_two_plus_three_way']['cv_mean']
    saturated_acc = results['saturated']['cv_mean']
    
    print(f"Performance improvements:")
    print(f"  Two-way interactions add: {two_way_acc - main_acc:+.3f}")
    print(f"  Three-way interactions add: {three_way_acc - two_way_acc:+.3f}")
    print(f"  Four-way interaction adds: {saturated_acc - three_way_acc:+.3f}")
    
    # Parameter efficiency
    print(f"\nParameter efficiency (accuracy per parameter):")
    for name, label in [('main_only', 'Main Effects'), 
                       ('main_plus_two_way', 'Main + Two-way'),
                       ('main_plus_two_plus_three_way', 'Main + Two + Three-way'),
                       ('saturated', 'Saturated')]:
        result = results[name]
        efficiency = result['cv_mean'] / result['n_params']
        print(f"  {label:<20}: {efficiency:.4f} accuracy/parameter")
    
    # ========== FINAL MODEL AND SUBMISSION ==========
    print("\n" + "="*50)
    print("FINAL MODEL TRAINING AND SUBMISSION")
    print("="*50)
    
    # Use best performing model
    best_features = results[best_model]['features']
    X_train_best = X_train[best_features]
    X_test_best = X_test[best_features]
    
    # Train final model
    lr_final = LogisticRegression(random_state=42, max_iter=1000)
    lr_final.fit(X_train_best, y)
    
    # Make predictions
    test_pred_proba = lr_final.predict_proba(X_test_best)[:, 1]
    test_pred = (test_pred_proba > 0.5).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_pred
    })
    
    submission.to_csv('submission_saturated_logistic.csv', index=False)
    
    # ========== SUBMISSION ANALYSIS ==========
    print(f"\nFinal Model: {best_model}")
    print(f"Features used: {len(best_features)}")
    print(f"CV Accuracy: {results[best_model]['cv_mean']:.3f} ± {results[best_model]['cv_std']:.3f}")
    print(f"Test predictions: {len(test_pred)} samples")
    print(f"Predicted survival rate: {test_pred.mean():.3f}")
    
    # Confidence analysis
    high_confidence = (test_pred_proba > 0.8) | (test_pred_proba < 0.2)
    print(f"High confidence predictions: {high_confidence.sum()} ({high_confidence.mean():.1%})")
    
    # Compare with previous models
    print(f"\nComparison with previous models:")
    print(f"• 4-feature Logistic (no interactions): ~80.8%")
    print(f"• 4-feature with manual interactions: 82.5%")
    print(f"• LightGBM/CatBoost 4-features: 78.23% (LB)")
    print(f"• Saturated logistic regression: {results[best_model]['cv_mean']:.1%}")
    
    # Overfitting analysis
    sample_to_param_ratio = len(y) / results[best_model]['n_params']
    print(f"\nModel complexity analysis:")
    print(f"• Sample size: {len(y)}")
    print(f"• Parameters: {results[best_model]['n_params']}")
    print(f"• Sample/parameter ratio: {sample_to_param_ratio:.1f}")
    
    if sample_to_param_ratio < 10:
        print("⚠️  Warning: Low sample-to-parameter ratio may indicate overfitting risk")
    elif sample_to_param_ratio < 20:
        print("⚡ Moderate complexity: Acceptable for this dataset size")
    else:
        print("✅ Conservative complexity: Low overfitting risk")
    
    print(f"\nSubmission file: submission_saturated_logistic.csv")
    
    if results[best_model]['cv_mean'] > 0.825:
        print("Expected performance: Likely competitive with gradient boosting models")
    else:
        print("Expected performance: Good baseline, may be outperformed by non-linear models")
    
    # ========== INTERPRETATION ==========
    print("\n" + "="*50)
    print("MODEL INTERPRETATION")
    print("="*50)
    
    print("Key insights from saturated model:")
    print("• All possible linear interactions captured")
    print("• Diminishing returns from higher-order interactions")
    print("• Most important terms:")
    
    # Show top 5 most important terms
    for i, (feature, coef, odds_ratio, abs_coef) in enumerate(coef_data[:5], 1):
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {i}. {feature}: OR={odds_ratio:.2f} ({direction} survival)")
    
    print(f"\nStatistical properties:")
    print(f"• Linear model with complete interaction structure")
    print(f"• Interpretable coefficients for policy insights")
    print(f"• Maximum complexity for linear relationships")

if __name__ == "__main__":
    main()