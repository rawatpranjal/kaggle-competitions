#!/usr/bin/env python3
"""
HOUSE PRICES UNIVARIATE ANALYSIS
================================
Simple baseline with log transform and univariate feature analysis.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train and test data"""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def analyze_target(train):
    """Analyze target variable distribution"""
    print("=" * 60)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 60)
    
    y = train['SalePrice']
    log_y = np.log1p(y)
    
    print(f"SalePrice Statistics:")
    print(f"  Count: {len(y):,}")
    print(f"  Mean: ${y.mean():,.0f}")
    print(f"  Median: ${y.median():,.0f}")
    print(f"  Std: ${y.std():,.0f}")
    print(f"  Min: ${y.min():,.0f}")
    print(f"  Max: ${y.max():,.0f}")
    print(f"  Skewness: {y.skew():.2f}")
    
    print(f"\nLog-transformed SalePrice:")
    print(f"  Mean: {log_y.mean():.3f}")
    print(f"  Std: {log_y.std():.3f}")
    print(f"  Skewness: {log_y.skew():.3f}")
    
    return log_y

def simple_missing_value_handling(df):
    """Handle missing values with simple strategies"""
    df_clean = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Fill categorical missing values with mode or 'Unknown'
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col].fillna(mode_val[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)
    
    return df_clean

def encode_categorical_features(train_df, test_df):
    """Simple label encoding for categorical features"""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Fit on combined data to ensure same encoding
        combined_values = pd.concat([train_df[col], test_df[col]], ignore_index=True)
        le.fit(combined_values.astype(str))
        
        train_encoded[col] = le.transform(train_df[col].astype(str))
        test_encoded[col] = le.transform(test_df[col].astype(str))
        
        encoders[col] = le
    
    return train_encoded, test_encoded, encoders

def univariate_feature_analysis(X, y, feature_names):
    """Test each feature individually with different models"""
    print("\n" + "=" * 80)
    print("UNIVARIATE FEATURE ANALYSIS")
    print("=" * 80)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    print(f"{'Feature':<25} {'Linear R²':<12} {'Ridge R²':<12} {'RF R²':<12} {'Missing %':<10}")
    print("-" * 80)
    
    for i, feature in enumerate(feature_names):
        if feature == 'SalePrice':  # Skip target
            continue
            
        # Get feature data
        X_single = X[[feature]].values.reshape(-1, 1)
        
        # Calculate missing percentage
        missing_pct = 100 * pd.Series(X_single.flatten()).isnull().sum() / len(X_single)
        
        try:
            # Linear Regression
            lr = LinearRegression()
            lr_scores = cross_val_score(lr, X_single, y, cv=cv, scoring='r2')
            lr_mean = np.mean(lr_scores)
            
            # Ridge Regression
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge_scores = cross_val_score(ridge, X_single, y, cv=cv, scoring='r2')
            ridge_mean = np.mean(ridge_scores)
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_scores = cross_val_score(rf, X_single, y, cv=cv, scoring='r2')
            rf_mean = np.mean(rf_scores)
            
            results.append({
                'feature': feature,
                'linear_r2': lr_mean,
                'ridge_r2': ridge_mean,
                'rf_r2': rf_mean,
                'missing_pct': missing_pct
            })
            
            print(f"{feature:<25} {lr_mean:8.3f}     {ridge_mean:8.3f}     {rf_mean:8.3f}     {missing_pct:6.1f}%")
            
        except Exception as e:
            print(f"{feature:<25} ERROR: {str(e)[:30]}")
            continue
    
    return pd.DataFrame(results)

def analyze_top_features(results_df, X, y, top_n=20):
    """Analyze the top performing features in detail"""
    print(f"\n" + "=" * 60)
    print(f"TOP {top_n} FEATURES ANALYSIS")
    print("=" * 60)
    
    # Sort by RF R² (usually most robust)
    top_features = results_df.nlargest(top_n, 'rf_r2')
    
    print("Top features by Random Forest R²:")
    print(f"{'Rank':<5} {'Feature':<25} {'RF R²':<10} {'Linear R²':<12} {'Missing %':<10}")
    print("-" * 65)
    
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:<5} {row['feature']:<25} {row['rf_r2']:6.3f}     {row['linear_r2']:8.3f}     {row['missing_pct']:6.1f}%")
    
    return top_features

def create_baseline_models(X, y, feature_names):
    """Create baseline models with different feature sets"""
    print(f"\n" + "=" * 60)
    print("BASELINE MODEL COMPARISON")
    print("=" * 60)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0, random_state=42),
        'Ridge (α=10)': Ridge(alpha=10.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    print(f"{'Model':<20} {'CV R²':<10} {'CV RMSE':<12} {'Features':<10}")
    print("-" * 55)
    
    baseline_results = []
    
    for model_name, model in models.items():
        try:
            # R² scores
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            r2_mean = np.mean(r2_scores)
            
            # RMSE scores (need to convert from neg_mean_squared_error)
            rmse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            rmse_mean = np.sqrt(-np.mean(rmse_scores))
            
            baseline_results.append({
                'model': model_name,
                'cv_r2': r2_mean,
                'cv_rmse': rmse_mean,
                'features': X.shape[1]
            })
            
            print(f"{model_name:<20} {r2_mean:6.3f}     {rmse_mean:8.0f}      {X.shape[1]:<10}")
            
        except Exception as e:
            print(f"{model_name:<20} ERROR: {str(e)[:20]}")
    
    return pd.DataFrame(baseline_results)

def create_submission(X_train, X_test, y_train, test_ids):
    """Create simple submission using best baseline model"""
    print(f"\n" + "=" * 50)
    print("CREATING SUBMISSION")
    print("=" * 50)
    
    # Use Ridge regression as baseline (usually robust)
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions (remember to inverse log transform)
    log_predictions = model.predict(X_test)
    predictions = np.expm1(log_predictions)  # Inverse of log1p
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission.to_csv('submissions/baseline_ridge.csv', index=False)
    
    print(f"Baseline Ridge submission created:")
    print(f"  File: submissions/baseline_ridge.csv")
    print(f"  Samples: {len(submission)}")
    print(f"  Price range: ${predictions.min():,.0f} - ${predictions.max():,.0f}")
    print(f"  Price median: ${np.median(predictions):,.0f}")
    
    return submission

def main():
    print("=" * 60)
    print("HOUSE PRICES UNIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Load data
    train, test = load_data()
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Analyze target variable
    log_y = analyze_target(train)
    
    # Handle missing values
    print(f"\nHandling missing values...")
    train_clean = simple_missing_value_handling(train)
    test_clean = simple_missing_value_handling(test)
    
    missing_before = train.isnull().sum().sum()
    missing_after = train_clean.isnull().sum().sum()
    print(f"Missing values: {missing_before} → {missing_after}")
    
    # Encode categorical features
    print(f"Encoding categorical features...")
    X_train_raw = train_clean.drop(['Id', 'SalePrice'], axis=1)
    X_test_raw = test_clean.drop(['Id'], axis=1)
    
    X_train_encoded, X_test_encoded, encoders = encode_categorical_features(X_train_raw, X_test_raw)
    
    print(f"Encoded features: {X_train_encoded.shape[1]}")
    
    # Univariate analysis
    results_df = univariate_feature_analysis(X_train_encoded, log_y, X_train_encoded.columns)
    
    # Analyze top features
    top_features = analyze_top_features(results_df, X_train_encoded, log_y)
    
    # Baseline models with all features
    baseline_results = create_baseline_models(X_train_encoded, log_y, X_train_encoded.columns)
    
    # Save results
    results_df.to_csv('univariate_results.csv', index=False)
    baseline_results.to_csv('baseline_results.csv', index=False)
    
    # Create submission
    test_ids = test['Id'].values
    submission = create_submission(X_train_encoded, X_test_encoded, log_y, test_ids)
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Results saved:")
    print(f"  univariate_results.csv - Individual feature performance")
    print(f"  baseline_results.csv - Baseline model comparison")
    print(f"  submissions/baseline_ridge.csv - Ready for submission")

if __name__ == "__main__":
    main()