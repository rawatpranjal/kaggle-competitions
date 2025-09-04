#!/usr/bin/env python3
"""
HOUSE PRICES CORRELATION ANALYSIS
=================================
Find correlations between features to identify unique information sources.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train data"""
    train = pd.read_csv('data/train.csv')
    return train

def encode_for_correlation(train):
    """Simple encoding for correlation analysis"""
    train_encoded = train.copy()
    
    # Encode categorical variables with label encoding
    categorical_cols = train.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        train_encoded[col] = le.fit_transform(train[col].astype(str))
    
    # Fill missing values with median/mode
    numeric_cols = train_encoded.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if train_encoded[col].isnull().sum() > 0:
            train_encoded[col].fillna(train_encoded[col].median(), inplace=True)
    
    return train_encoded

def analyze_feature_correlations(train_encoded):
    """Analyze correlations between all features"""
    print("=" * 70)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Remove Id and target for feature correlations
    features = train_encoded.drop(['Id', 'SalePrice'], axis=1)
    
    # Calculate correlation matrix
    corr_matrix = features.corr()
    
    print(f"Analyzing correlations between {len(features.columns)} features")
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print(f"\nHigh correlations (|r| > 0.7): {len(high_corr_pairs)} pairs")
    print(f"{'Feature 1':<20} {'Feature 2':<20} {'Correlation':<12}")
    print("-" * 55)
    
    for pair in high_corr_pairs:
        print(f"{pair['feature1']:<20} {pair['feature2']:<20} {pair['correlation']:8.3f}")
    
    return corr_matrix, high_corr_pairs

def identify_correlated_clusters(high_corr_pairs):
    """Group features into correlation clusters"""
    print(f"\n" + "=" * 50)
    print("CORRELATION CLUSTERS")
    print("=" * 50)
    
    # Build graph of correlations
    from collections import defaultdict
    
    graph = defaultdict(set)
    all_features = set()
    
    for pair in high_corr_pairs:
        f1, f2 = pair['feature1'], pair['feature2']
        graph[f1].add(f2)
        graph[f2].add(f1)
        all_features.update([f1, f2])
    
    # Find connected components (clusters)
    visited = set()
    clusters = []
    
    def dfs(node, cluster):
        if node in visited:
            return
        visited.add(node)
        cluster.append(node)
        for neighbor in graph[node]:
            dfs(neighbor, cluster)
    
    for feature in all_features:
        if feature not in visited:
            cluster = []
            dfs(feature, cluster)
            if len(cluster) > 1:
                clusters.append(cluster)
    
    print(f"Found {len(clusters)} correlation clusters:")
    
    for i, cluster in enumerate(clusters, 1):
        print(f"\nCluster {i} ({len(cluster)} features):")
        for feature in sorted(cluster):
            print(f"  {feature}")
    
    return clusters

def analyze_target_correlations_by_cluster(train_encoded, clusters):
    """Analyze target correlations within each cluster"""
    print(f"\n" + "=" * 60)
    print("TARGET CORRELATIONS BY CLUSTER")
    print("=" * 60)
    
    target = np.log1p(train_encoded['SalePrice'])  # Log transform
    
    cluster_analysis = []
    
    for i, cluster in enumerate(clusters, 1):
        print(f"\nCluster {i}:")
        print(f"{'Feature':<20} {'Target Corr':<12} {'Best in Cluster':<15}")
        print("-" * 50)
        
        cluster_corrs = []
        for feature in cluster:
            corr = train_encoded[feature].corr(target)
            cluster_corrs.append({'feature': feature, 'correlation': corr})
        
        # Sort by absolute correlation with target
        cluster_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        best_feature = cluster_corrs[0]['feature']
        
        for j, item in enumerate(cluster_corrs):
            is_best = "★ BEST" if j == 0 else ""
            print(f"{item['feature']:<20} {item['correlation']:8.3f}     {is_best:<15}")
        
        cluster_analysis.append({
            'cluster_id': i,
            'features': cluster,
            'best_feature': best_feature,
            'best_correlation': cluster_corrs[0]['correlation'],
            'feature_correlations': cluster_corrs
        })
    
    return cluster_analysis

def find_unique_information_sources(train_encoded, clusters):
    """Identify features with unique information (low correlation with others)"""
    print(f"\n" + "=" * 60)
    print("UNIQUE INFORMATION SOURCES")
    print("=" * 60)
    
    features = train_encoded.drop(['Id', 'SalePrice'], axis=1)
    target = np.log1p(train_encoded['SalePrice'])
    
    # Features already in clusters
    clustered_features = set()
    for cluster in clusters:
        clustered_features.update(cluster)
    
    # Analyze remaining features
    unique_features = []
    
    for feature in features.columns:
        if feature not in clustered_features:
            target_corr = train_encoded[feature].corr(target)
            
            # Check max correlation with any other feature
            max_corr = 0
            for other_feature in features.columns:
                if other_feature != feature:
                    corr = abs(train_encoded[feature].corr(train_encoded[other_feature]))
                    max_corr = max(max_corr, corr)
            
            unique_features.append({
                'feature': feature,
                'target_correlation': target_corr,
                'max_feature_correlation': max_corr
            })
    
    # Sort by target correlation strength
    unique_features.sort(key=lambda x: abs(x['target_correlation']), reverse=True)
    
    print(f"Features with unique information (not in correlation clusters):")
    print(f"{'Feature':<20} {'Target Corr':<12} {'Max Feat Corr':<15}")
    print("-" * 50)
    
    for item in unique_features:
        print(f"{item['feature']:<20} {item['target_correlation']:8.3f}     {item['max_feature_correlation']:8.3f}")
    
    return unique_features

def create_feature_selection_summary(cluster_analysis, unique_features):
    """Create summary of best features from each source"""
    print(f"\n" + "=" * 60)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 60)
    
    selected_features = []
    
    # Add best feature from each cluster
    print("Best features from correlation clusters:")
    for cluster in cluster_analysis:
        selected_features.append({
            'feature': cluster['best_feature'],
            'source': f"Cluster {cluster['cluster_id']}",
            'target_correlation': cluster['best_correlation']
        })
        
        print(f"  {cluster['best_feature']} (Cluster {cluster['cluster_id']}, r={cluster['best_correlation']:.3f})")
    
    # Add top unique features (target correlation > 0.3)
    print(f"\nTop unique features (|r| > 0.3):")
    for item in unique_features:
        if abs(item['target_correlation']) > 0.3:
            selected_features.append({
                'feature': item['feature'],
                'source': 'Unique',
                'target_correlation': item['target_correlation']
            })
            
            print(f"  {item['feature']} (r={item['target_correlation']:.3f})")
    
    # Sort final selection by target correlation
    selected_features.sort(key=lambda x: abs(x['target_correlation']), reverse=True)
    
    print(f"\n" + "=" * 40)
    print("FINAL FEATURE SELECTION")
    print("=" * 40)
    print(f"{'Feature':<20} {'Source':<12} {'Target Corr':<12}")
    print("-" * 45)
    
    for item in selected_features:
        print(f"{item['feature']:<20} {item['source']:<12} {item['target_correlation']:8.3f}")
    
    print(f"\nTotal selected features: {len(selected_features)}")
    
    return selected_features

def main():
    print("=" * 60)
    print("HOUSE PRICES CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Load and encode data
    train = load_data()
    train_encoded = encode_for_correlation(train)
    
    print(f"Dataset: {train_encoded.shape}")
    print(f"Features: {len(train_encoded.columns) - 2}")  # Exclude Id and SalePrice
    
    # Analyze feature correlations
    corr_matrix, high_corr_pairs = analyze_feature_correlations(train_encoded)
    
    # Identify correlation clusters
    clusters = identify_correlated_clusters(high_corr_pairs)
    
    # Analyze target correlations by cluster
    cluster_analysis = analyze_target_correlations_by_cluster(train_encoded, clusters)
    
    # Find unique information sources
    unique_features = find_unique_information_sources(train_encoded, clusters)
    
    # Create feature selection summary
    selected_features = create_feature_selection_summary(cluster_analysis, unique_features)
    
    # Save results
    corr_matrix.to_csv('feature_correlation_matrix.csv')
    
    selected_df = pd.DataFrame(selected_features)
    selected_df.to_csv('selected_features.csv', index=False)
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df.to_csv('high_correlations.csv', index=False)
    
    print(f"\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Files saved:")
    print("  feature_correlation_matrix.csv - Full correlation matrix")
    print("  selected_features.csv - Recommended feature selection")
    print("  high_correlations.csv - High correlation pairs")

if __name__ == "__main__":
    main()