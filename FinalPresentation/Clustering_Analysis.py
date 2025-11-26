"""
UNSUPERVISED CLUSTERING ANALYSIS - PRICERUNNER DATASET

Problem Statement:
    Discover natural groupings (clusters) in product data based on features
    WITHOUT using predefined labels. This is unsupervised learning.

Dataset:
    - 35,311 product samples
    - Features: Product Title (text), Merchant ID (categorical)
    - Ground Truth: Cluster ID (used for validation)

Methods:
    1. K-Means Clustering: Partitions data into k clusters
    2. DBSCAN Clustering: Finds density-based clusters (any shape)
    3. Hierarchical Clustering: Creates dendrogram of cluster hierarchy

Evaluation:
    - Silhouette Score: How similar objects are to their cluster vs. other clusters
    - Davies-Bouldin Index: Lower is better (cluster cohesion)
    - Calinski-Harabasz Index: Higher is better (cluster separation)
    - Adjusted Rand Index: Compare to ground truth (if available)

References:
    - Unsupervised Learning: ML without labeled data
    - Clustering: Grouping similar items together
    - Feature Engineering: Text vectorization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score, adjusted_rand_score)
import warnings
warnings.filterwarnings('ignore')

def get_user_input():
    """Collect user configuration"""
    print("\n" + "=" * 70)
    print("UNSUPERVISED CLUSTERING - CONFIGURATION")
    print("=" * 70)
    
    config = {}
    
    # K-Means clusters
    print("\n[K-MEANS CLUSTERING]")
    print("  Suggested: 5-10 clusters")
    print("  Range: 2-20")
    while True:
        try:
            user_input = input("  Enter number of clusters (or press Enter for 5): ").strip()
            if user_input == "":
                config['n_clusters'] = 5
            else:
                config['n_clusters'] = int(user_input)
                if not (2 <= config['n_clusters'] <= 20):
                    print("  ERROR: Please enter a value between 2 and 20")
                    continue
            print(f"  -> Selected: {config['n_clusters']} clusters")
            break
        except ValueError:
            print("  ERROR: Please enter a valid integer")
    
    # Feature selection
    print("\n[FEATURE SELECTION]")
    print("  Available options:")
    print("    1. Product Title + Merchant ID (recommended)")
    print("    2. Product Title only")
    print("    3. Merchant ID only (categorical)")
    while True:
        try:
            user_input = input("  Select option (or press Enter for 1): ").strip()
            if user_input == "":
                config['feature_choice'] = '1'
            else:
                if user_input not in ['1', '2', '3']:
                    print("  ERROR: Please enter 1, 2, or 3")
                    continue
                config['feature_choice'] = user_input
            
            if config['feature_choice'] == '1':
                print("  -> Selected: Product Title + Merchant ID")
            elif config['feature_choice'] == '2':
                print("  -> Selected: Product Title only")
            else:
                print("  -> Selected: Merchant ID only")
            break
        except ValueError:
            print("  ERROR: Please enter a valid option")
    
    # DBSCAN epsilon
    print("\n[DBSCAN CLUSTERING]")
    print("  Suggested: 0.5 (epsilon - distance parameter)")
    print("  Range: 0.1-2.0")
    while True:
        try:
            user_input = input("  Enter epsilon value (or press Enter for 0.5): ").strip()
            if user_input == "":
                config['eps'] = 0.5
            else:
                config['eps'] = float(user_input)
                if not (0.1 <= config['eps'] <= 2.0):
                    print("  ERROR: Please enter a value between 0.1 and 2.0")
                    continue
            print(f"  -> Selected: epsilon = {config['eps']}")
            break
        except ValueError:
            print("  ERROR: Please enter a valid number")
    
    # Run comparison
    print("\n[ALGORITHM SELECTION]")
    print("  Available options:")
    print("    1. K-Means only")
    print("    2. DBSCAN only")
    print("    3. Both (comparison)")
    while True:
        try:
            user_input = input("  Select option (or press Enter for 3): ").strip()
            if user_input == "":
                config['algorithm_choice'] = '3'
            else:
                if user_input not in ['1', '2', '3']:
                    print("  ERROR: Please enter 1, 2, or 3")
                    continue
                config['algorithm_choice'] = user_input
            
            if config['algorithm_choice'] == '1':
                print("  -> Selected: K-Means clustering")
            elif config['algorithm_choice'] == '2':
                print("  -> Selected: DBSCAN clustering")
            else:
                print("  -> Selected: Both algorithms (comparison)")
            break
        except ValueError:
            print("  ERROR: Please enter a valid option")
    
    return config

def load_and_prepare_data(config):
    """Load data and prepare features"""
    csv_file = 'pricerunner_aggregate.csv'
    
    print("\n" + "=" * 70)
    print("DATA LOADING AND PREPARATION")
    print("=" * 70)
    
    # Read data
    df = pd.read_csv(csv_file)
    print(f"\n[OK] Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} columns")
    
    # Data verification
    print("\nData Verification:")
    print(f"  - Missing values: {df.isnull().sum().sum()} total")
    
    # Extract features and ground truth
    product_titles = df['Product Title'].astype(str)
    merchant_ids = df[' Merchant ID']
    cluster_labels = df[' Cluster ID']  # Ground truth labels
    
    print(f"\n[OK] Data extracted:")
    print(f"  - Product titles: {len(product_titles)} samples")
    print(f"  - Merchant IDs: {merchant_ids.nunique()} unique merchants")
    print(f"  - Clusters (ground truth): {cluster_labels.nunique()} unique clusters")
    
    # Vectorize Product Title using CountVectorizer
    print(f"\n[OK] Vectorizing Product Title text...")
    vectorizer = CountVectorizer(max_features=100, lowercase=True, stop_words='english')
    X_title = vectorizer.fit_transform(product_titles)
    print(f"  - Generated {X_title.shape[1]} text features (word counts)")
    
    # Encode Merchant ID
    le_merchant = LabelEncoder()
    X_merchant = le_merchant.fit_transform(merchant_ids).reshape(-1, 1)
    print(f"  - Merchant ID encoded: {X_merchant.shape[1]} feature")
    
    # Combine features based on choice
    if config['feature_choice'] == '1':
        # Convert to dense, combine, convert back if needed for sklearn clustering
        X_title_dense = X_title.toarray()
        X = np.hstack([X_title_dense, X_merchant])
        features_used = "Product Title + Merchant ID"
    elif config['feature_choice'] == '2':
        X = X_title.toarray()  # Convert sparse to dense
        features_used = "Product Title"
    else:
        X = X_merchant
        features_used = "Merchant ID"
    
    print(f"\n[OK] Using {config['feature_choice']} feature(s): {features_used}")
    print(f"  - Total features: {X.shape[1]}")
    print(f"  - Total samples: {X.shape[0]:,}")
    
    return X, cluster_labels, features_used

def display_data_summary(X, cluster_labels, features_used):
    """Display data summary"""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset shape: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Features used: {features_used}")
    
    print("\nGround Truth Cluster Distribution (Cluster IDs):")
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique_clusters, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {cluster_id}: {count:6,} samples ({percentage:5.2f}%)")

def train_kmeans(X, config, cluster_labels):
    """Train K-Means clustering"""
    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  - Number of clusters: {config['n_clusters']}")
    print(f"  - Random state: 42 (reproducibility)")
    print(f"  - Algorithm: k-means++")
    
    # Train K-Means
    kmeans = KMeans(n_clusters=config['n_clusters'], random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    print(f"\n[OK] K-Means model trained successfully")
    
    # Cluster distribution
    print(f"\nK-Means Cluster Distribution (discovered clusters):")
    unique_kmeans, counts_kmeans = np.unique(kmeans_labels, return_counts=True)
    for cluster_id, count in zip(unique_kmeans, counts_kmeans):
        percentage = (count / len(kmeans_labels)) * 100
        print(f"  Cluster {cluster_id}: {count:6,} samples ({percentage:5.2f}%)")
    
    # Evaluation metrics
    print(f"\n[OK] Computing clustering quality metrics...")
    
    silhouette = silhouette_score(X, kmeans_labels)
    davies_bouldin = davies_bouldin_score(X, kmeans_labels)
    calinski_harabasz = calinski_harabasz_score(X, kmeans_labels)
    adjusted_rand = adjusted_rand_score(cluster_labels, kmeans_labels)
    
    print(f"\nClustering Quality Metrics:")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    print(f"    (Range: -1 to 1, higher is better)")
    print(f"    Interpretation: How similar objects are to their cluster")
    print(f"    {silhouette:.4f} = ", end="")
    if silhouette > 0.5:
        print("Good clustering [OK]")
    elif silhouette > 0.3:
        print("Reasonable clustering")
    else:
        print("Weak clustering")
    
    print(f"\n  - Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"    (Lower is better, range: 0 to infinity)")
    print(f"    Interpretation: Ratio of within-cluster to between-cluster distance")
    
    print(f"\n  - Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    print(f"    (Higher is better)")
    print(f"    Interpretation: Ratio of between-cluster to within-cluster variance")
    
    print(f"\n  - Adjusted Rand Index (vs. ground truth): {adjusted_rand:.4f}")
    print(f"    (Range: -1 to 1, higher is better)")
    print(f"    Interpretation: Agreement with known cluster labels")
    
    # Visualizations
    print(f"\n[OK] Generating K-Means visualizations...")
    
    # Cluster size comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: K-Means cluster distribution
    ax1 = axes[0]
    kmeans_unique = np.unique(kmeans_labels)
    kmeans_counts = [np.sum(kmeans_labels == c) for c in kmeans_unique]
    
    ax1.bar(range(len(kmeans_counts)), kmeans_counts, alpha=0.8, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Cluster ID', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('K-Means Cluster Distribution (discovered)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(kmeans_counts)))
    ax1.set_xticklabels(kmeans_unique)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(kmeans_counts):
        pct = count / len(kmeans_labels) * 100
        ax1.text(i, count + 100, f'{pct:.1f}%', ha='center', fontsize=9)
    
    # Plot 2: Metrics visualization
    ax2 = axes[1]
    metrics = ['Silhouette', 'Calinski-\nHarabasz\n(÷100)', 'Adjusted\nRand']
    values = [silhouette, calinski_harabasz/100, adjusted_rand]
    colors = ['green' if silhouette > 0.5 else 'orange' if silhouette > 0.3 else 'red']
    colors.extend(['steelblue', 'steelblue'])
    
    ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('K-Means Quality Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(values) * 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (metric, value) in enumerate(zip(metrics, values)):
        ax2.text(i, value + max(values)*0.03, f'{value:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('kmeans_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] kmeans_analysis.png saved")
    
    return kmeans, kmeans_labels, silhouette, davies_bouldin, calinski_harabasz, adjusted_rand

def train_dbscan(X, config, cluster_labels):
    """Train DBSCAN clustering"""
    print("\n" + "=" * 70)
    print("DBSCAN CLUSTERING")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  - Epsilon (radius): {config['eps']}")
    print(f"  - Min samples: 5 (default)")
    print(f"  - Metric: euclidean distance")
    
    # Normalize data for DBSCAN (important for distance-based clustering)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    # Train DBSCAN
    dbscan = DBSCAN(eps=config['eps'], min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    print(f"\n[OK] DBSCAN model trained successfully")
    
    # Cluster distribution
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"\nDBSCAN Cluster Distribution (discovered clusters):")
    print(f"  - Number of clusters: {n_clusters_dbscan}")
    print(f"  - Noise points: {n_noise:,} ({n_noise/len(dbscan_labels)*100:.2f}%)")
    
    for cluster_id in sorted(set(dbscan_labels)):
        if cluster_id == -1:
            continue
        count = list(dbscan_labels).count(cluster_id)
        percentage = (count / len(dbscan_labels)) * 100
        print(f"  Cluster {cluster_id}: {count:6,} samples ({percentage:5.2f}%)")
    
    # Evaluation metrics (excluding noise points for Silhouette)
    print(f"\n[OK] Computing clustering quality metrics...")
    
    if len(set(dbscan_labels)) > 1:  # At least 2 clusters
        # For Silhouette, use only non-noise points
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 0 and len(set(dbscan_labels[non_noise_mask])) > 1:
            X_non_noise = X_scaled[non_noise_mask]
            labels_non_noise = dbscan_labels[non_noise_mask]
            silhouette = silhouette_score(X_non_noise, labels_non_noise)
        else:
            silhouette = -1
        
        davies_bouldin = davies_bouldin_score(X_scaled[non_noise_mask], 
                                              dbscan_labels[non_noise_mask]) if np.sum(non_noise_mask) > 0 else -1
        calinski_harabasz = calinski_harabasz_score(X_scaled[non_noise_mask], 
                                                     dbscan_labels[non_noise_mask]) if np.sum(non_noise_mask) > 0 else -1
    else:
        silhouette = davies_bouldin = calinski_harabasz = -1
    
    # Adjusted Rand (includes noise as separate cluster)
    adjusted_rand = adjusted_rand_score(cluster_labels, dbscan_labels)
    
    print(f"\nClustering Quality Metrics:")
    if silhouette >= 0:
        print(f"  - Silhouette Score: {silhouette:.4f}")
    else:
        print(f"  - Silhouette Score: N/A")
    
    if davies_bouldin >= 0:
        print(f"  - Davies-Bouldin Index: {davies_bouldin:.4f}")
    else:
        print(f"  - Davies-Bouldin Index: N/A")
    
    if calinski_harabasz >= 0:
        print(f"  - Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    else:
        print(f"  - Calinski-Harabasz Index: N/A")
    
    print(f"  - Adjusted Rand Index (vs. ground truth): {adjusted_rand:.4f}")
    print(f"  - Noise Points: {n_noise:,} ({n_noise/len(dbscan_labels)*100:.2f}%)")
    
    # Visualizations
    print(f"\n[OK] Generating DBSCAN visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cluster size comparison
    ax1 = axes[0]
    dbscan_unique = sorted(set(dbscan_labels))
    dbscan_counts = [list(dbscan_labels).count(c) for c in dbscan_unique]
    
    colors = ['red' if c == -1 else 'steelblue' for c in dbscan_unique]
    ax1.bar(range(len(dbscan_unique)), dbscan_counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Cluster ID', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('DBSCAN Cluster Distribution (Red = Noise)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(dbscan_unique)))
    ax1.set_xticklabels(dbscan_unique)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Info summary
    ax2 = axes[1]
    ax2.axis('off')
    
    info_text = f"""
DBSCAN Configuration:
  • Epsilon: {config['eps']}
  • Min Samples: 5

Results:
  • Clusters Found: {n_clusters_dbscan}
  • Noise Points: {n_noise:,}
  • Adjusted Rand: {adjusted_rand:.4f}
  
Interpretation:
  • Density-based clustering
  • Can find clusters of any shape
  • Marks outliers as noise
  • Good for spatial data
"""
    
    ax2.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dbscan_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] dbscan_analysis.png saved")
    
    return dbscan, dbscan_labels, silhouette, davies_bouldin, calinski_harabasz, adjusted_rand

def display_summary(config, kmeans_results=None, dbscan_results=None):
    """Display comprehensive summary"""
    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("=" * 70)
    
    if config['algorithm_choice'] in ['1', '3'] and kmeans_results:
        kmeans, kmeans_labels, sil_k, db_k, ch_k, ar_k = kmeans_results
        
        print(f"\n[K-MEANS CLUSTERING]")
        print(f"  Number of clusters: {config['n_clusters']}")
        print(f"  Silhouette Score: {sil_k:.4f} (similarity to own cluster)")
        print(f"  Davies-Bouldin Index: {db_k:.4f} (within/between distance ratio)")
        print(f"  Calinski-Harabasz Index: {ch_k:.4f} (between/within variance ratio)")
        print(f"  Adjusted Rand Index: {ar_k:.4f} (agreement with ground truth)")
        
        if sil_k > 0.5:
            quality = "Good clustering [OK]"
        elif sil_k > 0.3:
            quality = "Reasonable clustering"
        else:
            quality = "Weak clustering"
        
        print(f"  Assessment: {quality}")
    
    if config['algorithm_choice'] in ['2', '3'] and dbscan_results:
        dbscan, dbscan_labels, sil_d, db_d, ch_d, ar_d = dbscan_results
        
        print(f"\n[DBSCAN CLUSTERING]")
        print(f"  Epsilon: {config['eps']}")
        
        if sil_d >= 0:
            print(f"  Silhouette Score: {sil_d:.4f}")
        else:
            print(f"  Silhouette Score: N/A")
        
        if db_d >= 0:
            print(f"  Davies-Bouldin Index: {db_d:.4f}")
        else:
            print(f"  Davies-Bouldin Index: N/A")
        
        if ch_d >= 0:
            print(f"  Calinski-Harabasz Index: {ch_d:.4f}")
        else:
            print(f"  Calinski-Harabasz Index: N/A")
        
        print(f"  Adjusted Rand Index: {ar_d:.4f} (agreement with ground truth)")
        print(f"  Assessment: Density-based approach (can find irregular shapes)")
    
    if config['algorithm_choice'] == '3' and kmeans_results and dbscan_results:
        print(f"\n[ALGORITHM COMPARISON]")
        print(f"  K-Means Adjusted Rand: {ar_k:.4f}")
        print(f"  DBSCAN Adjusted Rand: {ar_d:.4f}")
        
        if ar_k > ar_d:
            print(f"  Winner: K-Means (better agreement with ground truth)")
        elif ar_d > ar_k:
            print(f"  Winner: DBSCAN (better agreement with ground truth)")
        else:
            print(f"  Tie: Both algorithms perform equally")
    
    print("\n" + "=" * 70)
    print("Output files generated:")
    if config['algorithm_choice'] in ['1', '3']:
        print("  - kmeans_analysis.png")
    if config['algorithm_choice'] in ['2', '3']:
        print("  - dbscan_analysis.png")
    print("=" * 70 + "\n")

def display_clustering_concepts():
    """Display educational information about clustering"""
    print("\n" + "=" * 70)
    print("UNSUPERVISED CLUSTERING - CONCEPTS")
    print("=" * 70)
    
    print("""
[WHAT IS CLUSTERING?]
  Unsupervised Learning: No labeled data needed
  Goal: Find natural groups (clusters) in data
  Use Cases: Customer segmentation, document grouping, image compression

[K-MEANS CLUSTERING]
  Algorithm:
    1. Initialize k random cluster centers
    2. Assign each point to nearest center (Euclidean distance)
    3. Recalculate centers as mean of assigned points
    4. Repeat steps 2-3 until convergence
  
  Pros: Simple, fast, works well for spherical clusters
  Cons: Must specify k in advance, can get stuck in local optima
  
  Interpretation:
    - Silhouette Score > 0.5: Good clustering
    - Silhouette Score 0.3-0.5: Reasonable clustering
    - Silhouette Score < 0.3: Weak clustering

[DBSCAN CLUSTERING]
  Algorithm:
    1. For each point, find all neighbors within epsilon distance
    2. If point has >= min_samples neighbors, it's a core point
    3. Form clusters from connected core points
    4. Remaining points are noise/outliers
  
  Pros: Finds any shape, detects outliers, no k needed
  Cons: Sensitive to epsilon parameter, struggles with varying density
  
  Interpretation:
    - Noise points: Outliers or sparse regions
    - Multiple clusters: Natural groupings found
    - Single cluster: Epsilon too large or data too uniform

[EVALUATION METRICS]
  Silhouette Score: -1 (bad) to 1 (perfect)
    - Measures cluster cohesion and separation
    - High = well-defined clusters
  
  Davies-Bouldin Index: 0 (perfect) to infinity
    - Ratio of within-cluster to between-cluster distance
    - Lower = better separated clusters
  
  Calinski-Harabasz Index: 0 to infinity
    - Ratio of between-cluster to within-cluster variance
    - Higher = better separated clusters
  
  Adjusted Rand Index: -1 to 1
    - Compares clustering to known labels
    - 1 = perfect agreement
    - 0 = random agreement
    - Useful for validation with ground truth

[GROUND TRUTH COMPARISON]
  Even though clustering is unsupervised, we can compare results to:
  - Predefined categories (if available)
  - Domain knowledge
  - Adjusted Rand Index measures agreement
  - Can validate if clustering makes sense
""")

def main():
    """Main execution"""
    print("\n")
    print("=" * 70)
    print("  UNSUPERVISED CLUSTERING ANALYSIS - PRICERUNNER DATASET".center(70))
    print("  K-Means vs. DBSCAN Clustering Comparison".center(70))
    print("=" * 70)
    
    run_again = True
    while run_again:
        # Display concepts
        display_clustering_concepts()
        
        # Get user configuration
        config = get_user_input()
        
        # Load and prepare data
        X, cluster_labels, features_used = load_and_prepare_data(config)
        
        # Display data summary
        display_data_summary(X, cluster_labels, features_used)
        
        # Train models based on choice
        kmeans_results = None
        dbscan_results = None
        
        if config['algorithm_choice'] in ['1', '3']:
            kmeans_results = train_kmeans(X, config, cluster_labels)
        
        if config['algorithm_choice'] in ['2', '3']:
            dbscan_results = train_dbscan(X, config, cluster_labels)
        
        # Display summary
        display_summary(config, kmeans_results, dbscan_results)
        
        # Ask if user wants to run again
        print("\n" + "=" * 70)
        while True:
            response = input("Would you like to run with different parameters? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                run_again = True
                print("\n" + "-" * 70 + "\n")
                break
            elif response in ['no', 'n']:
                run_again = False
                print("\n[OK] Analysis complete.")
                print("=" * 70 + "\n")
                break
            else:
                print("Please enter 'yes' or 'no'")

if __name__ == "__main__":
    main()
