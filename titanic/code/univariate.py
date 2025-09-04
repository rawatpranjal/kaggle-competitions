#!/usr/bin/env python3
"""
SIMPLE UNIVARIATE ANALYSIS
===========================
Test each feature individually including decile dummies for continuous variables
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def test_feature(X, y, feature_name):
    """Test a single feature's predictive power"""
    # Handle NaN
    mask = ~pd.isna(X)
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(np.unique(X_clean)) < 2:
        return None
    
    # Reshape for sklearn
    if len(X_clean.shape) == 1:
        X_clean = X_clean.reshape(-1, 1)
    
    try:
        # Simple logistic regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        # 5-fold CV accuracy
        cv_scores = cross_val_score(lr, X_clean, y_clean, cv=5, scoring='accuracy')
        cv_accuracy = np.mean(cv_scores)
        
        # Train for AUC
        lr.fit(X_clean, y_clean)
        y_proba = lr.predict_proba(X_clean)[:, 1]
        auc = roc_auc_score(y_clean, y_proba)
        
        # Calculate survival rate for binary features
        if len(np.unique(X_clean)) == 2:
            survival_rate = y_clean[X_clean.flatten() == 1].mean()
        else:
            survival_rate = None
        
        return {
            'feature': feature_name,
            'cv_accuracy': cv_accuracy,
            'auc': auc,
            'survival_rate': survival_rate,
            'n_unique': len(np.unique(X_clean)),
            'n_samples': len(X_clean)
        }
    except:
        return None

def create_features(df):
    """Create all features to test"""
    features = {}
    
    # ========== ORIGINAL NUMERIC FEATURES ==========
    features['Pclass'] = df['Pclass'].values
    features['Age'] = df['Age'].values
    features['SibSp'] = df['SibSp'].values
    features['Parch'] = df['Parch'].values
    features['Fare'] = df['Fare'].values
    
    # ========== BINARY FEATURES ==========
    features['Sex_Male'] = (df['Sex'] == 'male').astype(int).values
    features['Sex_Female'] = (df['Sex'] == 'female').astype(int).values
    
    # ========== EMBARKED DUMMIES ==========
    features['Embarked_S'] = (df['Embarked'] == 'S').astype(int).values
    features['Embarked_C'] = (df['Embarked'] == 'C').astype(int).values
    features['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int).values
    
    # ========== CABIN FEATURES ==========
    features['HasCabin'] = df['Cabin'].notna().astype(int).values
    
    # Cabin deck dummies
    cabin_deck = df['Cabin'].str[0]
    for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']:
        features[f'Deck_{deck}'] = (cabin_deck == deck).astype(int).values
    
    # ========== AGE DECILES ==========
    age_filled = df['Age'].fillna(df['Age'].median())
    age_deciles = pd.qcut(age_filled, 10, labels=False, duplicates='drop')
    for i in range(10):
        features[f'Age_Decile_{i+1}'] = (age_deciles == i).astype(int).values
    
    # Age groups
    features['Age_Child'] = (age_filled < 12).astype(int).values
    features['Age_Teen'] = ((age_filled >= 12) & (age_filled < 18)).astype(int).values
    features['Age_Adult'] = ((age_filled >= 18) & (age_filled < 60)).astype(int).values
    features['Age_Senior'] = (age_filled >= 60).astype(int).values
    
    # ========== FARE DECILES ==========
    fare_filled = df['Fare'].fillna(df['Fare'].median())
    fare_deciles = pd.qcut(fare_filled, 10, labels=False, duplicates='drop')
    for i in range(10):
        features[f'Fare_Decile_{i+1}'] = (fare_deciles == i).astype(int).values
    
    # Fare groups
    features['Fare_0'] = (fare_filled == 0).astype(int).values
    features['Fare_Low'] = ((fare_filled > 0) & (fare_filled < 10)).astype(int).values
    features['Fare_Med'] = ((fare_filled >= 10) & (fare_filled < 30)).astype(int).values
    features['Fare_High'] = (fare_filled >= 30).astype(int).values
    
    # ========== FAMILY FEATURES ==========
    family_size = df['SibSp'] + df['Parch'] + 1
    features['FamilySize'] = family_size.values
    features['IsAlone'] = (family_size == 1).astype(int).values
    features['SmallFamily'] = ((family_size > 1) & (family_size <= 4)).astype(int).values
    features['LargeFamily'] = (family_size > 4).astype(int).values
    
    # SibSp dummies
    for i in range(9):
        features[f'SibSp_{i}'] = (df['SibSp'] == i).astype(int).values
    
    # Parch dummies
    for i in range(10):
        features[f'Parch_{i}'] = (df['Parch'] == i).astype(int).values
    
    # ========== PCLASS DUMMIES ==========
    features['Pclass_1'] = (df['Pclass'] == 1).astype(int).values
    features['Pclass_2'] = (df['Pclass'] == 2).astype(int).values
    features['Pclass_3'] = (df['Pclass'] == 3).astype(int).values
    
    # ========== NAME/TITLE FEATURES ==========
    # Extract title
    title = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Common titles as dummies
    for t in ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Col', 'Major', 'Capt']:
        features[f'Title_{t}'] = (title == t).astype(int).values
    
    # Title groups
    features['Title_Mr'] = (title == 'Mr').astype(int).values
    features['Title_Miss'] = title.isin(['Miss', 'Mlle', 'Ms']).astype(int).values
    features['Title_Mrs'] = title.isin(['Mrs', 'Mme']).astype(int).values
    features['Title_Master'] = (title == 'Master').astype(int).values
    features['Title_Military'] = title.isin(['Col', 'Major', 'Capt']).astype(int).values
    features['Title_Noble'] = title.isin(['Lady', 'Countess', 'Sir', 'Don', 'Dona', 'Jonkheer']).astype(int).values
    
    # Name characteristics
    features['Name_HasParentheses'] = df['Name'].str.contains(r'\(').astype(int).values
    features['NameLength'] = df['Name'].str.len().values
    
    # ========== TICKET FEATURES ==========
    features['Ticket_Numeric'] = df['Ticket'].str.isdigit().astype(int).values
    features['Ticket_HasLetters'] = df['Ticket'].str.contains('[A-Za-z]').astype(int).values
    
    # Ticket frequency
    ticket_counts = df['Ticket'].value_counts()
    features['Ticket_Alone'] = df['Ticket'].map(lambda x: ticket_counts[x] == 1).astype(int).values
    features['Ticket_Small'] = df['Ticket'].map(lambda x: 2 <= ticket_counts[x] <= 4).astype(int).values
    features['Ticket_Large'] = df['Ticket'].map(lambda x: ticket_counts[x] > 4).astype(int).values
    
    # ========== TRANSFORMATIONS ==========
    # Log transformations
    features['Age_Log'] = np.log1p(age_filled).values
    features['Fare_Log'] = np.log1p(fare_filled).values
    
    # Square root
    features['Age_Sqrt'] = np.sqrt(age_filled).values
    features['Fare_Sqrt'] = np.sqrt(fare_filled).values
    
    return features

def main():
    print("="*70)
    print("SIMPLE UNIVARIATE ANALYSIS")
    print("="*70)
    
    # Load data
    train = pd.read_csv('../data/train.csv')
    y = train['Survived'].values
    
    print(f"\nDataset: {len(train)} samples")
    print(f"Survival rate: {y.mean():.3f}")
    
    # Create all features
    print("\nCreating features...")
    features = create_features(train)
    print(f"Total features to test: {len(features)}")
    
    # Test each feature
    results = []
    print("\nTesting each feature:")
    print("-"*70)
    
    for feature_name, feature_values in features.items():
        result = test_feature(feature_values, y, feature_name)
        if result:
            results.append(result)
            if len(results) % 20 == 0:
                print(f"  Tested {len(results)} features...")
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cv_accuracy', ascending=False)
    
    # ========== RESULTS ==========
    print("\n" + "="*70)
    print("TOP 30 FEATURES BY CROSS-VALIDATION ACCURACY")
    print("="*70)
    
    for i, row in enumerate(results_df.head(30).itertuples(), 1):
        survival_info = f"(survival: {row.survival_rate:.3f})" if row.survival_rate else ""
        print(f"{i:2}. {row.feature:25} CV: {row.cv_accuracy:.3f}  AUC: {row.auc:.3f} {survival_info}")
    
    # ========== ANALYSIS BY TYPE ==========
    print("\n" + "="*70)
    print("ANALYSIS BY FEATURE TYPE")
    print("="*70)
    
    # Best Age feature
    print("\nBEST AGE FEATURES:")
    age_features = results_df[results_df['feature'].str.contains('Age')]
    for row in age_features.head(5).itertuples():
        print(f"  {row.feature:20} CV: {row.cv_accuracy:.3f}")
    
    # Best Fare feature
    print("\nBEST FARE FEATURES:")
    fare_features = results_df[results_df['feature'].str.contains('Fare')]
    for row in fare_features.head(5).itertuples():
        print(f"  {row.feature:20} CV: {row.cv_accuracy:.3f}")
    
    # Best Title feature
    print("\nBEST TITLE FEATURES:")
    title_features = results_df[results_df['feature'].str.contains('Title')]
    for row in title_features.head(5).itertuples():
        print(f"  {row.feature:20} CV: {row.cv_accuracy:.3f}")
    
    # Compare deciles
    print("\nAGE DECILE PERFORMANCE:")
    for i in range(1, 11):
        decile = results_df[results_df['feature'] == f'Age_Decile_{i}']
        if not decile.empty:
            row = decile.iloc[0]
            print(f"  Decile {i:2}: CV: {row['cv_accuracy']:.3f}  Survival: {row['survival_rate']:.3f}")
    
    print("\nFARE DECILE PERFORMANCE:")
    for i in range(1, 11):
        decile = results_df[results_df['feature'] == f'Fare_Decile_{i}']
        if not decile.empty:
            row = decile.iloc[0]
            print(f"  Decile {i:2}: CV: {row['cv_accuracy']:.3f}  Survival: {row['survival_rate']:.3f}")
    
    # ========== CORRELATION ANALYSIS ==========
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS - FINDING UNIQUE INFORMATION")
    print("="*70)
    
    # Create feature matrix for top features
    top_features = results_df.head(20)['feature'].tolist()
    
    feature_matrix = []
    feature_names = []
    
    for feature_name in top_features:
        if feature_name in features:
            feature_values = features[feature_name]
            # Fill NaN with median for correlation calculation
            if np.any(pd.isna(feature_values)):
                feature_values = pd.Series(feature_values).fillna(pd.Series(feature_values).median()).values
            feature_matrix.append(feature_values)
            feature_names.append(feature_name)
    
    # Calculate correlation matrix
    feature_df = pd.DataFrame(np.array(feature_matrix).T, columns=feature_names)
    corr_matrix = feature_df.corr().abs()
    
    print("\nHIGH CORRELATIONS (>0.8) - REDUNDANT FEATURES:")
    print("-"*50)
    redundant_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.8:
                feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                redundant_pairs.append((feat1, feat2, corr_val))
                print(f"  {feat1:20} <-> {feat2:20} r={corr_val:.3f}")
    
    if not redundant_pairs:
        print("  No highly correlated features found")
    
    print(f"\nMODERATE CORRELATIONS (0.5-0.8) - RELATED FEATURES:")
    print("-"*50)
    moderate_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if 0.5 <= corr_val <= 0.8:
                feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                moderate_pairs.append((feat1, feat2, corr_val))
                print(f"  {feat1:20} <-> {feat2:20} r={corr_val:.3f}")
    
    # Find unique information sources
    print(f"\nUNIQUE INFORMATION SOURCES:")
    print("-"*50)
    
    # Group highly correlated features
    feature_groups = {}
    processed = set()
    
    for feat1, feat2, corr_val in redundant_pairs:
        if feat1 not in processed and feat2 not in processed:
            group_name = f"Group_{feat1.split('_')[0]}"
            if group_name not in feature_groups:
                feature_groups[group_name] = []
            feature_groups[group_name].extend([feat1, feat2])
            processed.update([feat1, feat2])
    
    # Add ungrouped features
    for feat in feature_names:
        if feat not in processed:
            feature_groups[feat] = [feat]
    
    print(f"Found {len(feature_groups)} unique information sources:")
    for group_name, group_features in feature_groups.items():
        if len(group_features) > 1:
            # Find best performing feature in group
            group_results = results_df[results_df['feature'].isin(group_features)]
            best_feat = group_results.iloc[0]['feature']
            best_acc = group_results.iloc[0]['cv_accuracy']
            print(f"  {group_name:15} ({len(group_features)} features) -> Best: {best_feat} ({best_acc:.3f})")
        else:
            feat_result = results_df[results_df['feature'] == group_features[0]]
            if not feat_result.empty:
                acc = feat_result.iloc[0]['cv_accuracy']
                print(f"  {group_features[0]:15} (unique) -> CV: {acc:.3f}")
    
    # Feature diversity analysis
    print(f"\nFEATURE DIVERSITY ANALYSIS:")
    print("-"*50)
    
    # Calculate average correlation with target for each feature
    target_correlations = []
    for feature_name in feature_names:
        feature_values = feature_df[feature_name].values
        # Remove any remaining NaN
        mask = ~pd.isna(feature_values) & ~pd.isna(y)
        if mask.sum() > 10:  # Need enough samples
            corr_with_target, _ = pearsonr(feature_values[mask], y[mask])
            target_correlations.append(abs(corr_with_target))
        else:
            target_correlations.append(0)
    
    # Find features with low inter-correlation but decent target correlation
    diverse_features = []
    for i, feat in enumerate(feature_names):
        avg_inter_corr = corr_matrix.iloc[i].drop(feat).mean()
        target_corr = target_correlations[i]
        if avg_inter_corr < 0.3 and target_corr > 0.2:  # Low inter-correlation, decent target correlation
            diverse_features.append((feat, target_corr, avg_inter_corr))
    
    diverse_features.sort(key=lambda x: x[1], reverse=True)  # Sort by target correlation
    
    print("Features with unique information (low inter-correlation, decent target correlation):")
    for feat, target_corr, inter_corr in diverse_features[:10]:
        feat_result = results_df[results_df['feature'] == feat]
        if not feat_result.empty:
            cv_acc = feat_result.iloc[0]['cv_accuracy']
            print(f"  {feat:20} Target r={target_corr:.3f}, Inter r={inter_corr:.3f}, CV={cv_acc:.3f}")
    
    # Recommended feature set
    print(f"\nRECOMMENDED MINIMAL FEATURE SET:")
    print("-"*50)
    print("Based on performance and low redundancy:")
    
    recommended = []
    used_groups = set()
    
    for _, row in results_df.head(20).iterrows():
        feat = row['feature']
        feat_group = feat.split('_')[0] if '_' in feat else feat
        
        if feat_group not in used_groups:
            recommended.append(feat)
            used_groups.add(feat_group)
            if len(recommended) >= 10:  # Limit to top 10 diverse features
                break
    
    for i, feat in enumerate(recommended, 1):
        feat_result = results_df[results_df['feature'] == feat]
        cv_acc = feat_result.iloc[0]['cv_accuracy']
        print(f"  {i:2}. {feat:20} CV: {cv_acc:.3f}")
    
    # ========== SIMPLE LOGIT: SEX + CHILD + CABIN + INTERACTIONS ==========
    print("\n" + "="*70)
    print("SIMPLE LOGISTIC REGRESSION - SEX + CHILD + CABIN + INTERACTIONS")
    print("="*70)
    
    # Create feature dataframe with interactions
    feature_df = pd.DataFrame()
    feature_df['Sex_Male'] = pd.Series(features['Sex_Male'])
    feature_df['Age_Child'] = pd.Series(features['Age_Child'])
    feature_df['HasCabin'] = pd.Series(features['HasCabin'])
    
    # Create two-way interactions
    feature_df['Sex_Child'] = feature_df['Sex_Male'] * feature_df['Age_Child']
    feature_df['Sex_Cabin'] = feature_df['Sex_Male'] * feature_df['HasCabin']
    feature_df['Child_Cabin'] = feature_df['Age_Child'] * feature_df['HasCabin']
    
    # Create three-way interaction
    feature_df['Sex_Child_Cabin'] = feature_df['Sex_Male'] * feature_df['Age_Child'] * feature_df['HasCabin']
    
    # All features
    final_features = ['Sex_Male', 'Age_Child', 'HasCabin', 'Sex_Child', 'Sex_Cabin', 'Child_Cabin', 'Sex_Child_Cabin']
    X_final = feature_df[final_features]
    
    # Cross-validation for logistic regression
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_scores = cross_val_score(lr, X_final, y, cv=cv, scoring='accuracy')
    
    print(f"Features used: {final_features}")
    print(f"CV Accuracy: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")
    print(f"Individual folds: {[f'{s:.3f}' for s in lr_scores]}")
    
    # Fit model to see coefficients
    lr.fit(X_final, y)
    print("\nLogistic Regression Coefficients:")
    for feature, coef in zip(final_features, lr.coef_[0]):
        odds_ratio = np.exp(coef)
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {feature:18}: {coef:+.3f}  (OR={odds_ratio:.2f}, {direction} survival)")
    
    print(f"  Intercept: {lr.intercept_[0]:+.3f}")
    
    # Show survival rates by all 8 groups
    print("\nSurvival Rates by Group (M=Male, C=Child, Cab=Cabin):")
    groups = [
        (0, 0, 0, "Female Adult No-Cabin"),
        (0, 0, 1, "Female Adult Cabin"),
        (0, 1, 0, "Female Child No-Cabin"),
        (0, 1, 1, "Female Child Cabin"),
        (1, 0, 0, "Male Adult No-Cabin"),
        (1, 0, 1, "Male Adult Cabin"),
        (1, 1, 0, "Male Child No-Cabin"),
        (1, 1, 1, "Male Child Cabin")
    ]
    
    for male, child, cabin, label in groups:
        mask = ((feature_df['Sex_Male'] == male) & 
                (feature_df['Age_Child'] == child) & 
                (feature_df['HasCabin'] == cabin))
        count = mask.sum()
        if count > 0:
            survival_rate = y[mask].mean()
            print(f"  {label:20} (n={count:2}): {survival_rate:.3f}")
        else:
            print(f"  {label:20} (n= 0): N/A")
    
    # Model interpretation
    print("\nModel Interpretation:")
    print("  Baseline: Female Adults with No Cabin")
    print("  Main effects: Sex_Male, Age_Child, HasCabin")
    print("  Two-way interactions: Sex*Child, Sex*Cabin, Child*Cabin")
    print("  Three-way interaction: Sex*Child*Cabin")
    
    # Compare with simpler models
    sex_result = results_df[results_df['feature'] == 'Sex_Male'].iloc[0]
    child_result = results_df[results_df['feature'] == 'Age_Child'].iloc[0]
    cabin_result = results_df[results_df['feature'] == 'HasCabin'].iloc[0]
    
    print(f"\nComparison with Individual Features:")
    print(f"  Sex_Male alone: {sex_result['cv_accuracy']:.3f}")
    print(f"  Age_Child alone: {child_result['cv_accuracy']:.3f}")
    print(f"  HasCabin alone: {cabin_result['cv_accuracy']:.3f}")
    print(f"  Full interaction model: {lr_scores.mean():.3f}")
    print(f"  Improvement over Sex: {lr_scores.mean() - sex_result['cv_accuracy']:+.3f}")
    
    # Test simpler model for comparison
    simple_features = ['Sex_Male', 'Age_Child', 'HasCabin']
    X_simple = feature_df[simple_features]
    lr_simple_scores = cross_val_score(lr, X_simple, y, cv=cv, scoring='accuracy')
    
    print(f"\nSimple additive model (no interactions):")
    print(f"  CV Accuracy: {lr_simple_scores.mean():.3f} ± {lr_simple_scores.std():.3f}")
    print(f"  vs Full model: {lr_scores.mean() - lr_simple_scores.mean():+.3f}")
    
    # Sample size analysis
    print(f"\nSample Sizes by Group:")
    for male, child, cabin, label in groups:
        mask = ((feature_df['Sex_Male'] == male) & 
                (feature_df['Age_Child'] == child) & 
                (feature_df['HasCabin'] == cabin))
        count = mask.sum()
        print(f"  {label:20}: {count:3} samples")
    
    # ========== FIND UNIQUE RISK DIMENSION ==========
    print("\n" + "="*70)
    print("FINDING UNIQUE RISK DIMENSION - PREDICTION RESIDUAL ANALYSIS")
    print("="*70)
    
    # Get predictions from our best model
    y_pred_proba = lr.predict_proba(X_final)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate residuals (actual - predicted)
    residuals = y - y_pred_proba
    
    print(f"Model accuracy on full data: {(y == y_pred).mean():.3f}")
    print(f"Mean absolute residual: {np.abs(residuals).mean():.3f}")
    print(f"Residual std: {residuals.std():.3f}")
    
    # Find cases with largest positive residuals (survived when predicted low)
    # These are the "lucky survivors" our model missed
    high_pos_residuals = np.where(residuals > 0.4)[0]  # Much better than predicted
    high_neg_residuals = np.where(residuals < -0.4)[0]  # Much worse than predicted
    
    print(f"\nLucky Survivors (actual > predicted by >0.4): {len(high_pos_residuals)} cases")
    print(f"Unlucky Deaths (actual < predicted by >0.4): {len(high_neg_residuals)} cases")
    
    # Create residual features to test against all other features
    print(f"\nTesting residuals against all other features to find missing dimension...")
    
    # Test residuals against all features
    residual_correlations = []
    
    for feature_name, feature_values in features.items():
        if feature_name not in ['Sex_Male', 'Age_Child', 'HasCabin']:  # Skip features already in model
            # Handle NaN values
            mask = ~pd.isna(feature_values)
            if mask.sum() < 20:  # Skip features with too few values
                continue
                
            feature_clean = feature_values[mask]
            residual_clean = residuals[mask]
            
            if len(np.unique(feature_clean)) > 1:
                try:
                    # Calculate correlation with residuals
                    corr_coef, p_value = pearsonr(feature_clean, residual_clean)
                    
                    # Also test if feature explains residual variance
                    feature_reshaped = feature_clean.reshape(-1, 1)
                    lr_residual = LogisticRegression(max_iter=1000, random_state=42)
                    
                    # Use absolute residuals as target (magnitude of error)
                    abs_residuals = np.abs(residual_clean)
                    
                    # Convert to binary: large error vs small error
                    large_error = (abs_residuals > abs_residuals.median()).astype(int)
                    
                    if len(np.unique(large_error)) > 1:  # Make sure we have both classes
                        cv_scores_residual = cross_val_score(lr_residual, feature_reshaped, large_error, cv=3, scoring='accuracy')
                        residual_accuracy = np.mean(cv_scores_residual)
                        
                        residual_correlations.append({
                            'feature': feature_name,
                            'residual_corr': abs(corr_coef),
                            'p_value': p_value,
                            'error_prediction_acc': residual_accuracy,
                            'n_samples': len(feature_clean)
                        })
                except:
                    continue
    
    # Sort by residual correlation
    residual_df = pd.DataFrame(residual_correlations)
    if len(residual_df) > 0:
        residual_df = residual_df.sort_values('residual_corr', ascending=False)
    else:
        print("No valid residual correlations found")
    
    print(f"\nTOP FEATURES CORRELATED WITH MODEL RESIDUALS:")
    print("-" * 60)
    print("Feature                   Residual_Corr  Error_Pred_Acc  P_Value")
    print("-" * 60)
    
    for _, row in residual_df.head(15).iterrows():
        significance = "*" if row['p_value'] < 0.05 else " "
        print(f"{row['feature']:25} {row['residual_corr']:8.3f}    {row['error_prediction_acc']:8.3f}    {row['p_value']:7.3f}{significance}")
    
    # Test the top residual-correlated feature
    if len(residual_df) > 0:
        best_residual_feature = residual_df.iloc[0]['feature']
        print(f"\n" + "="*50)
        print(f"TESTING BEST RESIDUAL FEATURE: {best_residual_feature}")
        print("="*50)
        
        # Add this feature to our model
        feature_df[best_residual_feature] = pd.Series(features[best_residual_feature])
        
        # Test model with additional feature
        enhanced_features = final_features + [best_residual_feature]
        X_enhanced = feature_df[enhanced_features].fillna(0)  # Fill NaN with 0
        
        lr_enhanced = LogisticRegression(random_state=42, max_iter=1000)
        enhanced_scores = cross_val_score(lr_enhanced, X_enhanced, y, cv=cv, scoring='accuracy')
        
        print(f"Original model: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")
        print(f"Enhanced model: {enhanced_scores.mean():.3f} ± {enhanced_scores.std():.3f}")
        print(f"Improvement: {enhanced_scores.mean() - lr_scores.mean():+.3f}")
        
        # Analyze the feature in context of our groups
        print(f"\nAnalyzing {best_residual_feature} across our 8 groups:")
        
        for male, child, cabin, label in groups:
            mask = ((feature_df['Sex_Male'] == male) & 
                    (feature_df['Age_Child'] == child) & 
                    (feature_df['HasCabin'] == cabin))
            
            if mask.sum() > 0:
                group_feature_values = feature_df[best_residual_feature][mask]
                group_survival = y[mask]
                group_residuals = residuals[mask]
                
                # Skip if all NaN
                if not pd.isna(group_feature_values).all():
                    feature_mean = group_feature_values.mean()
                    residual_mean = group_residuals.mean()
                    print(f"  {label:20}: {best_residual_feature}={feature_mean:.3f}, residual={residual_mean:+.3f}")
        
        # Show some specific examples of lucky/unlucky cases
        print(f"\nEXAMPLES OF MODEL ERRORS:")
        print("-" * 40)
        
        # Lucky survivors (high positive residuals)
        if len(high_pos_residuals) > 0:
            print("LUCKY SURVIVORS (actual=1, predicted low):")
            for i in high_pos_residuals[:5]:  # Show first 5
                pred_prob = y_pred_proba[i]
                actual = y[i]
                male = feature_df['Sex_Male'].iloc[i]
                child = feature_df['Age_Child'].iloc[i]
                cabin = feature_df['HasCabin'].iloc[i]
                feature_val = feature_df[best_residual_feature].iloc[i]
                
                group_type = f"{'Male' if male else 'Female'} {'Child' if child else 'Adult'} {'Cabin' if cabin else 'No-Cabin'}"
                print(f"  {group_type:20} pred={pred_prob:.3f} actual={actual} {best_residual_feature}={feature_val}")
        
        # Unlucky deaths
        if len(high_neg_residuals) > 0:
            print("\nUNLUCKY DEATHS (actual=0, predicted high):")
            for i in high_neg_residuals[:5]:  # Show first 5
                pred_prob = y_pred_proba[i]
                actual = y[i]
                male = feature_df['Sex_Male'].iloc[i]
                child = feature_df['Age_Child'].iloc[i]
                cabin = feature_df['HasCabin'].iloc[i]
                feature_val = feature_df[best_residual_feature].iloc[i]
                
                group_type = f"{'Male' if male else 'Female'} {'Child' if child else 'Adult'} {'Cabin' if cabin else 'No-Cabin'}"
                print(f"  {group_type:20} pred={pred_prob:.3f} actual={actual} {best_residual_feature}={feature_val}")
    
    # Test embarked as a potential unique dimension
    print(f"\n" + "="*50)
    print("SPECIAL ANALYSIS: EMBARKED AS UNIQUE DIMENSION")
    print("="*50)
    
    # Look at survival by embarkation point within our groups
    embarked_feature = pd.Series(train['Embarked'].fillna('S'))  # Fill missing with S
    feature_df['Embarked_S'] = (embarked_feature == 'S').astype(int)
    feature_df['Embarked_C'] = (embarked_feature == 'C').astype(int)
    feature_df['Embarked_Q'] = (embarked_feature == 'Q').astype(int)
    
    print("Survival by Embarkation Port:")
    for port in ['S', 'C', 'Q']:
        port_mask = (embarked_feature == port)
        if port_mask.sum() > 0:
            port_survival = y[port_mask].mean()
            port_count = port_mask.sum()
            print(f"  Port {port}: {port_survival:.3f} ({port_count} passengers)")
    
    # Test if adding Embarked improves our model
    embarked_features = final_features + ['Embarked_C', 'Embarked_Q']  # Use C and Q (S is baseline)
    X_embarked = feature_df[embarked_features].fillna(0)
    
    lr_embarked = LogisticRegression(random_state=42, max_iter=1000)
    embarked_scores = cross_val_score(lr_embarked, X_embarked, y, cv=cv, scoring='accuracy')
    
    print(f"\nWith Embarked added:")
    print(f"Original model: {lr_scores.mean():.3f}")
    print(f"With Embarked:  {embarked_scores.mean():.3f}")
    print(f"Improvement: {embarked_scores.mean() - lr_scores.mean():+.3f}")
    
    # Final comprehensive model test
    print(f"\n" + "="*50)
    print("FINAL COMPREHENSIVE MODEL")
    print("="*50)
    
    # Combine best features from different dimensions
    final_comprehensive = final_features + [best_residual_feature, 'Embarked_C'] if len(residual_df) > 0 else final_features + ['Embarked_C']
    X_comprehensive = feature_df[final_comprehensive].fillna(0)
    
    lr_comprehensive = LogisticRegression(random_state=42, max_iter=1000)
    comprehensive_scores = cross_val_score(lr_comprehensive, X_comprehensive, y, cv=cv, scoring='accuracy')
    
    print(f"Features: {final_comprehensive}")
    print(f"CV Accuracy: {comprehensive_scores.mean():.3f} ± {comprehensive_scores.std():.3f}")
    print(f"vs Original: {comprehensive_scores.mean() - lr_scores.mean():+.3f}")
    print(f"vs Sex alone: {comprehensive_scores.mean() - sex_result['cv_accuracy']:+.3f}")
    
    # Save results
    results_df.to_csv('../simple_univariate_results.csv', index=False)
    corr_matrix.to_csv('../feature_correlations.csv')
    print(f"\n\nResults saved to simple_univariate_results.csv")
    print(f"Correlation matrix saved to feature_correlations.csv")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Features tested: {len(results_df)}")
    print(f"Features with CV > 0.65: {len(results_df[results_df['cv_accuracy'] > 0.65])}")
    print(f"Features with CV > 0.70: {len(results_df[results_df['cv_accuracy'] > 0.70])}")
    print(f"Best accuracy: {results_df['cv_accuracy'].max():.3f}")
    print(f"Median accuracy: {results_df['cv_accuracy'].median():.3f}")
    print(f"Unique information sources: {len(feature_groups)}")

if __name__ == "__main__":
    main()