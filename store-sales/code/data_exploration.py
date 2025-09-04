#!/usr/bin/env python3
"""
STORE SALES DATA EXPLORATION
============================
Initial exploration of the Favorita store sales dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all competition datasets"""
    print("Loading datasets...")
    
    train = pd.read_csv('data/train.csv', parse_dates=['date'])
    test = pd.read_csv('data/test.csv', parse_dates=['date'])
    stores = pd.read_csv('data/stores.csv')
    oil = pd.read_csv('data/oil.csv', parse_dates=['date'])
    holidays = pd.read_csv('data/holidays_events.csv', parse_dates=['date'])
    transactions = pd.read_csv('data/transactions.csv', parse_dates=['date'])
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")
    print(f"Stores: {stores.shape}")
    print(f"Oil: {oil.shape}")
    print(f"Holidays: {holidays.shape}")
    print(f"Transactions: {transactions.shape}")
    print(f"Sample submission: {sample_submission.shape}")
    
    return train, test, stores, oil, holidays, transactions, sample_submission

def explore_training_data(train):
    """Explore the training dataset"""
    print("\n" + "="*60)
    print("TRAINING DATA EXPLORATION")
    print("="*60)
    
    print(f"Date range: {train['date'].min()} to {train['date'].max()}")
    print(f"Total days: {(train['date'].max() - train['date'].min()).days + 1}")
    print(f"Unique stores: {train['store_nbr'].nunique()}")
    print(f"Unique product families: {train['family'].nunique()}")
    print(f"Total records: {len(train):,}")
    
    print(f"\nSales statistics:")
    print(train['sales'].describe())
    
    print(f"\nProduct families:")
    family_counts = train['family'].value_counts()
    print(f"Top 10 families by record count:")
    for i, (family, count) in enumerate(family_counts.head(10).items()):
        print(f"  {i+1:2d}. {family:<25} {count:>8,} records")
    
    print(f"\nPromotion statistics:")
    print(f"Records with promotions: {(train['onpromotion'] > 0).sum():,} ({(train['onpromotion'] > 0).mean()*100:.1f}%)")
    print(train['onpromotion'].describe())
    
    # Check for missing values
    print(f"\nMissing values:")
    missing = train.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"  {col}: {missing[col]:,} ({missing[col]/len(train)*100:.1f}%)")
    
    # Zero sales analysis
    zero_sales = (train['sales'] == 0).sum()
    print(f"\nZero sales records: {zero_sales:,} ({zero_sales/len(train)*100:.1f}%)")
    
    return family_counts

def explore_stores_data(stores):
    """Explore stores metadata"""
    print("\n" + "="*60)
    print("STORES DATA EXPLORATION")
    print("="*60)
    
    print(stores.info())
    print(f"\nStore distribution:")
    print(f"Cities: {stores['city'].nunique()}")
    print(f"States: {stores['state'].nunique()}")
    print(f"Store types: {stores['type'].nunique()}")
    print(f"Clusters: {stores['cluster'].nunique()}")
    
    print(f"\nStore type distribution:")
    print(stores['type'].value_counts())
    
    print(f"\nCluster distribution:")
    print(stores['cluster'].value_counts().sort_index())

def explore_external_data(oil, holidays, transactions):
    """Explore external datasets"""
    print("\n" + "="*60)
    print("EXTERNAL DATA EXPLORATION")
    print("="*60)
    
    # Oil data
    print("OIL PRICES:")
    print(f"Date range: {oil['date'].min()} to {oil['date'].max()}")
    print(f"Missing oil prices: {oil['dcoilwtico'].isnull().sum()} days")
    print(f"Oil price range: ${oil['dcoilwtico'].min():.2f} - ${oil['dcoilwtico'].max():.2f}")
    
    # Holidays data
    print(f"\nHOLIDAYS:")
    print(f"Date range: {holidays['date'].min()} to {holidays['date'].max()}")
    print(f"Holiday types:")
    print(holidays['type'].value_counts())
    print(f"\nLocale types:")
    print(holidays['locale'].value_counts())
    
    # Transactions data
    print(f"\nTRANSACTIONS:")
    print(f"Date range: {transactions['date'].min()} to {transactions['date'].max()}")
    print(f"Stores with transaction data: {transactions['store_nbr'].nunique()}")
    print(f"Transaction statistics:")
    print(transactions['transactions'].describe())

def analyze_time_patterns(train):
    """Analyze temporal patterns in sales"""
    print("\n" + "="*60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*60)
    
    # Add time features
    train_time = train.copy()
    train_time['year'] = train_time['date'].dt.year
    train_time['month'] = train_time['date'].dt.month
    train_time['day_of_week'] = train_time['date'].dt.dayofweek
    train_time['day_name'] = train_time['date'].dt.day_name()
    
    # Overall sales by year
    print("Sales by year:")
    yearly_sales = train_time.groupby('year')['sales'].agg(['sum', 'mean', 'count']).round(2)
    for year, data in yearly_sales.iterrows():
        print(f"  {year}: Total=${data['sum']:>12,.0f}, Mean=${data['mean']:>6.2f}, Records={data['count']:>8,}")
    
    # Sales by month
    print(f"\nSales by month:")
    monthly_sales = train_time.groupby('month')['sales'].mean().round(2)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, avg_sales in monthly_sales.items():
        print(f"  {month_names[month-1]:3s}: ${avg_sales:>8.2f}")
    
    # Sales by day of week
    print(f"\nSales by day of week:")
    daily_sales = train_time.groupby('day_name')['sales'].mean().round(2)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in day_order:
        if day in daily_sales.index:
            print(f"  {day:9s}: ${daily_sales[day]:>8.2f}")

def analyze_promotions_impact(train):
    """Analyze impact of promotions on sales"""
    print("\n" + "="*60)
    print("PROMOTION IMPACT ANALYSIS")
    print("="*60)
    
    # Sales with vs without promotions
    promo_sales = train[train['onpromotion'] > 0]['sales'].mean()
    no_promo_sales = train[train['onpromotion'] == 0]['sales'].mean()
    
    print(f"Average sales with promotions:    ${promo_sales:>10.2f}")
    print(f"Average sales without promotions: ${no_promo_sales:>10.2f}")
    print(f"Promotion lift:                   {(promo_sales/no_promo_sales - 1)*100:>7.1f}%")
    
    # Promotion intensity analysis
    promo_intensity = train[train['onpromotion'] > 0].copy()
    print(f"\nPromotion intensity distribution:")
    print(promo_intensity['onpromotion'].describe())

def analyze_test_data(test, train):
    """Analyze test data and prediction requirements"""
    print("\n" + "="*60)
    print("TEST DATA ANALYSIS")
    print("="*60)
    
    print(f"Test period: {test['date'].min()} to {test['date'].max()}")
    print(f"Prediction days: {(test['date'].max() - test['date'].min()).days + 1}")
    print(f"Records to predict: {len(test):,}")
    
    # Check if test stores/families match training
    train_stores = set(train['store_nbr'].unique())
    test_stores = set(test['store_nbr'].unique())
    train_families = set(train['family'].unique())
    test_families = set(test['family'].unique())
    
    print(f"\nStore coverage:")
    print(f"  Training stores: {len(train_stores)}")
    print(f"  Test stores: {len(test_stores)}")
    print(f"  Stores in both: {len(train_stores & test_stores)}")
    print(f"  New stores in test: {len(test_stores - train_stores)}")
    
    print(f"\nFamily coverage:")
    print(f"  Training families: {len(train_families)}")
    print(f"  Test families: {len(test_families)}")
    print(f"  Families in both: {len(train_families & test_families)}")
    print(f"  New families in test: {len(test_families - train_families)}")
    
    # Gap between train and test
    last_train_date = train['date'].max()
    first_test_date = test['date'].min()
    gap_days = (first_test_date - last_train_date).days - 1
    
    print(f"\nData gap:")
    print(f"  Last training date: {last_train_date}")
    print(f"  First test date: {first_test_date}")
    print(f"  Gap: {gap_days} days")

def main():
    """Main exploration function"""
    print("=" * 60)
    print("STORE SALES DATA EXPLORATION")
    print("=" * 60)
    
    # Load data
    train, test, stores, oil, holidays, transactions, sample_submission = load_data()
    
    # Explore each dataset
    family_counts = explore_training_data(train)
    explore_stores_data(stores)
    explore_external_data(oil, holidays, transactions)
    
    # Analyze patterns
    analyze_time_patterns(train)
    analyze_promotions_impact(train)
    analyze_test_data(test, train)
    
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    
    print("Key insights:")
    print("- Time series forecasting problem with 15-day prediction horizon")
    print("- 54 stores, 33 product families, 4+ years of training data")
    print("- External factors: oil prices, holidays, promotions affect sales")
    print("- Strong seasonal and weekly patterns in sales")
    print("- Promotions provide significant sales lift")
    print("- No missing stores/families in test set")
    
    print("\nNext steps:")
    print("1. Time series feature engineering (lags, rolling statistics)")
    print("2. External factor integration (oil, holidays)")
    print("3. Store and family clustering analysis") 
    print("4. Baseline forecasting model development")

if __name__ == "__main__":
    main()