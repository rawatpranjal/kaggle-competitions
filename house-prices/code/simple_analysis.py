#!/usr/bin/env python3
"""
HOUSE PRICES SIMPLE ANALYSIS
============================
One-way analysis with correlations and means, then simple linear model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train and test data"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def analyze_target(train):
    """Analyze target variable"""
    print("=" * 50)
    print("TARGET ANALYSIS")
    print("=" * 50)
    
    y = train['SalePrice']
    log_y = np.log1p(y)
    
    print(f"SalePrice:")
    print(f"  Mean: ${y.mean():,.0f}")
    print(f"  Median: ${y.median():,.0f}")
    print(f"  Std: ${y.std():,.0f}")
    print(f"  Skewness: {y.skew():.2f}")
    
    print(f"\nLog(SalePrice):")
    print(f"  Mean: {log_y.mean():.3f}")
    print(f"  Std: {log_y.std():.3f}")
    print(f"  Skewness: {log_y.skew():.3f}")
    
    return log_y

def analyze_numeric_features(train, log_y):
    """Analyze numeric features with simple correlations"""
    print("\n" + "=" * 60)
    print("NUMERIC FEATURES CORRELATION ANALYSIS")
    print("=" * 60)
    
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('Id')
    numeric_cols.remove('SalePrice')
    
    correlations = []
    
    print(f"{'Feature':<20} {'Correlation':<12} {'Missing %':<10} {'Mean':<12}")
    print("-" * 60)
    
    for col in numeric_cols:
        corr = train[col].corr(log_y)
        missing_pct = 100 * train[col].isnull().sum() / len(train)
        mean_val = train[col].mean()
        
        correlations.append({
            'feature': col,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'missing_pct': missing_pct,
            'mean': mean_val
        })
        
        print(f"{col:<20} {corr:8.3f}     {missing_pct:6.1f}%     {mean_val:8.0f}")
    
    # Sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    print(f"\nTop 10 Numeric Features by |Correlation|:")
    print(f"{'Feature':<20} {'Correlation':<12}")
    print("-" * 35)
    
    for _, row in corr_df.head(10).iterrows():
        print(f"{row['feature']:<20} {row['correlation']:8.3f}")
    
    return corr_df

def analyze_categorical_features(train, log_y):
    """Analyze categorical features with group means"""
    print("\n" + "=" * 70)
    print("CATEGORICAL FEATURES MEAN ANALYSIS")
    print("=" * 70)
    
    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    category_results = []
    
    for col in categorical_cols:
        # Calculate group means
        group_means = train.groupby(col)['SalePrice'].agg(['mean', 'count']).reset_index()
        group_means = group_means.sort_values('mean', ascending=False)
        
        # Calculate range of means (max - min)
        mean_range = group_means['mean'].max() - group_means['mean'].min()
        
        # Calculate missing percentage
        missing_pct = 100 * train[col].isnull().sum() / len(train)
        
        # Number of unique categories
        n_categories = train[col].nunique()
        
        category_results.append({
            'feature': col,
            'mean_range': mean_range,
            'n_categories': n_categories,
            'missing_pct': missing_pct,
            'highest_mean': group_means.iloc[0]['mean'],
            'lowest_mean': group_means.iloc[-1]['mean']
        })
        
        print(f"\n{col} (Missing: {missing_pct:.1f}%, Categories: {n_categories}):")
        print(f"  Price Range: ${group_means['mean'].min():,.0f} - ${group_means['mean'].max():,.0f}")
        
        # Show top 5 and bottom 5 categories
        print("  Highest prices:")
        for _, row in group_means.head(3).iterrows():
            print(f"    {row[col]}: ${row['mean']:,.0f} (n={row['count']})")
        
        print("  Lowest prices:")
        for _, row in group_means.tail(3).iterrows():
            print(f"    {row[col]}: ${row['mean']:,.0f} (n={row['count']})")
    
    # Sort by mean range
    cat_df = pd.DataFrame(category_results)
    cat_df = cat_df.sort_values('mean_range', ascending=False)
    
    print(f"\n" + "=" * 50)
    print("CATEGORICAL FEATURES RANKED BY PRICE RANGE")
    print("=" * 50)
    print(f"{'Feature':<20} {'Price Range':<15} {'Categories':<12}")
    print("-" * 50)
    
    for _, row in cat_df.iterrows():
        print(f"{row['feature']:<20} ${row['mean_range']:8,.0f}      {row['n_categories']:5d}")
    
    return cat_df

def simple_linear_model(train, numeric_features, categorical_features):
    """Build simple linear model and interpret coefficients"""
    print("\n" + "=" * 60)
    print("SIMPLE LINEAR MODEL")
    print("=" * 60)
    
    # Prepare features
    X_features = []
    feature_names = []
    
    # Add top numeric features (top 10 by correlation)
    top_numeric = numeric_features.head(10)['feature'].tolist()
    for col in top_numeric:
        if train[col].isnull().sum() == 0:  # Only use features without missing values for simplicity
            X_features.append(train[col].values)
            feature_names.append(col)
    
    # Add top categorical features (encode as dummy variables)
    top_categorical = categorical_features.head(5)['feature'].tolist()
    for col in top_categorical:
        if train[col].isnull().sum() == 0:  # Only use features without missing values
            # Create dummy variables
            dummies = pd.get_dummies(train[col], prefix=col)
            for dummy_col in dummies.columns[:-1]:  # Drop last category to avoid collinearity
                X_features.append(dummies[dummy_col].values)
                feature_names.append(dummy_col)
    
    # Combine features
    if len(X_features) == 0:
        print("No features available for modeling")
        return
    
    X = np.column_stack(X_features)
    y = np.log1p(train['SalePrice'])
    
    print(f"Model features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Fit linear model
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X, y, cv=cv, scoring='r2')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"CV R²: {cv_mean:.3f} ± {cv_std:.3f}")
    
    # Interpret coefficients
    print(f"\nModel Coefficients:")
    print(f"{'Feature':<25} {'Coefficient':<12} {'Impact':<15}")
    print("-" * 55)
    
    coefficients = list(zip(feature_names, lr.coef_))
    coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, coef in coefficients:
        # Calculate impact: 1 unit change -> % change in price
        pct_impact = (np.exp(coef) - 1) * 100
        impact_str = f"{pct_impact:+.1f}%"
        
        print(f"{feature:<25} {coef:8.3f}     {impact_str:<15}")
    
    print(f"\nIntercept: {lr.intercept_:.3f}")
    print(f"Baseline price (intercept): ${np.exp(lr.intercept_):,.0f}")
    
    # Feature importance by absolute coefficient
    print(f"\nTop 10 Most Important Features:")
    print(f"{'Feature':<25} {'|Coefficient|':<15}")
    print("-" * 42)
    
    for i, (feature, coef) in enumerate(coefficients[:10], 1):
        print(f"{i:2d}. {feature:<22} {abs(coef):8.3f}")
    
    return lr, X, y, feature_names

def main():
    print("=" * 50)
    print("HOUSE PRICES SIMPLE ANALYSIS")
    print("=" * 50)
    
    # Load data
    train, test = load_data()
    print(f"Train shape: {train.shape}")
    
    # Analyze target
    log_y = analyze_target(train)
    
    # Analyze numeric features
    numeric_results = analyze_numeric_features(train, log_y)
    
    # Analyze categorical features
    categorical_results = analyze_categorical_features(train, log_y)
    
    # Simple linear model
    model_results = simple_linear_model(train, numeric_results, categorical_results)
    
    # Save results
    numeric_results.to_csv('numeric_correlations.csv', index=False)
    categorical_results.to_csv('categorical_means.csv', index=False)
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Files saved:")
    print("  numeric_correlations.csv")
    print("  categorical_means.csv")

if __name__ == "__main__":
    main()