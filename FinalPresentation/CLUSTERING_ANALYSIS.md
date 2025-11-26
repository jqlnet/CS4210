# Unsupervised Clustering Analysis - PriceRunner Dataset

## 📊 Project Overview

**Objective:** Discover natural groupings in product data using unsupervised learning, without relying on predefined labels.

**Dataset:** 35,311 PriceRunner products
**Features:** Product Title (100 text features) + Merchant ID (1 categorical feature)
**Methods:** K-Means Clustering vs. DBSCAN Clustering

**Key Question:** Can we automatically discover meaningful clusters that align with the predefined Cluster IDs?

---

## 🎯 Problem Statement

### Supervised vs. Unsupervised Learning

**Supervised Learning** (Classification - Earlier Projects):
- Training data: Features + Labels
- Goal: Predict labels for new samples
- Example: "Classify product into category"
- Assumption: Labels are given

**Unsupervised Learning** (Clustering - This Project):
- Training data: Features ONLY (no labels)
- Goal: Discover hidden structure/patterns
- Example: "Find natural product groupings"
- Assumption: No labels available

### Real-World Motivation

Customer segmentation, document grouping, anomaly detection, product recommendation all require finding natural clusters without predefined categories.

---

## 📈 Clustering Methods Compared

### K-Means Clustering

**Algorithm Concept:**
```
1. Choose k (number of clusters)
2. Randomly initialize k cluster centers
3. Assign each point to nearest center
4. Recalculate centers as mean of assigned points
5. Repeat steps 3-4 until convergence
```

**Characteristics:**
- **Partitioning:** Each point belongs to exactly one cluster
- **Distance metric:** Euclidean distance
- **Strengths:** Simple, fast, scalable
- **Weaknesses:** Must know k in advance, assumes spherical clusters

**Test Results (k=5):**
```
Silhouette Score: 0.6965 (Good clustering)
Davies-Bouldin Index: 0.4534 (Well-separated clusters)
Calinski-Harabasz Index: 460,933.29 (Strong cluster structure)
Cluster Sizes:
  - Cluster 0: 6,401 samples (18.13%)
  - Cluster 1: 1,635 samples (4.63%)
  - Cluster 2: 7,449 samples (21.10%)
  - Cluster 3: 8,448 samples (23.92%)
  - Cluster 4: 11,378 samples (32.22%)
```

**Interpretation:**
- 0.6965 Silhouette score indicates good clustering
- Clusters are well-separated from each other
- However, Adjusted Rand Index (0.0001) shows poor agreement with ground truth
- Suggests discovered clusters differ from predefined Cluster IDs

---

### DBSCAN Clustering

**Algorithm Concept:**
```
1. Define epsilon (radius) and min_samples (minimum neighbors)
2. For each point:
   - If it has >= min_samples neighbors within epsilon distance:
     → Mark as core point, form/extend cluster
   - Otherwise:
     → Mark as noise (outlier)
3. Connect core points that are density-reachable
```

**Characteristics:**
- **Density-based:** Finds clusters based on density, not distance to center
- **Flexible shapes:** Can find arbitrary-shaped clusters
- **Noise detection:** Marks outliers as noise points
- **Strengths:** No k needed, finds any shape, detects outliers
- **Weaknesses:** Sensitive to epsilon, struggles with varying densities

**Test Results (epsilon=0.5):**
```
Silhouette Score: 0.9226 (Excellent clustering)
Davies-Bouldin Index: 0.1072 (Highly separated clusters)
Calinski-Harabasz Index: 12,071.87 (Very strong structure)
Clusters Found: 1,281 clusters (!)
Noise Points: 13,957 (39.53%)
```

**Interpretation:**
- 0.9226 Silhouette score indicates excellent quality
- Discovers very fine-grained clusters (1,281 clusters)
- 39.53% of data marked as noise/outliers
- DBSCAN finds density-based structure at finer granularity
- With epsilon=0.5, considers even small product variations as separate clusters

---

## 🔍 Key Findings

### Finding 1: Both Algorithms Succeed at Clustering (Quality-wise)

| Metric | K-Means | DBSCAN | Assessment |
|--------|---------|--------|-----------|
| Silhouette Score | 0.6965 | 0.9226 | DBSCAN better (more compact) |
| Davies-Bouldin | 0.4534 | 0.1072 | DBSCAN better (more separated) |
| Calinski-Harabasz | 460,933 | 12,072 | K-Means better (but scores not directly comparable) |

**Conclusion:** Both algorithms produce high-quality clusters. DBSCAN shows tighter, more well-defined clustering.

---

### Finding 2: Both Fail to Recover Predefined Clusters

| Metric | K-Means | DBSCAN |
|--------|---------|--------|
| **Adjusted Rand Index** | **0.0001** | **0.0002** |
| Interpretation | Almost no agreement | Almost no agreement |

**Why?**
- Ground truth has Cluster IDs based on specific product matching
- K-Means discovers different natural groupings based on text similarity + merchant
- DBSCAN's fine-grained clustering doesn't align with broad categories

**Lesson:** Unsupervised clustering discovers DATA STRUCTURE, not necessarily predefined labels.

---

### Finding 3: Algorithm Choice Affects Interpretation

**K-Means (5 clusters):**
- Provides interpretable, fixed number of groups
- Useful for quick segmentation
- Assumes clusters are roughly equal size

**DBSCAN (1,281 clusters):**
- Discovers natural granularity
- Groups only very similar products
- Marks 40% as outliers (unique products)

**Choice Depends On:**
- Do you need a specific number of segments? → K-Means
- Do you want to identify outliers? → DBSCAN
- Do you want to explore natural structure? → DBSCAN
- Do you need fast computation? → K-Means

---

## 📊 Clustering Quality Metrics Explained

### Silhouette Score (-1 to 1, higher is better)

**Formula:**
```
For each point i:
  a(i) = avg distance to points in same cluster
  b(i) = min avg distance to points in other clusters
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

Silhouette Score = mean(s(i))
```

**Interpretation:**
- **1.0** = Perfect clustering
- **0.5+** = Good clustering
- **0.3-0.5** = Reasonable clustering
- **0.0** = No clear clustering
- **Negative** = Samples in wrong clusters

**Results:**
- K-Means: 0.6965 → Good clustering
- DBSCAN: 0.9226 → Excellent clustering

---

### Davies-Bouldin Index (0 to infinity, lower is better)

**Formula:**
```
For each cluster i:
  R_i = max(s_i + s_j) / d(c_i, c_j)  over all j != i
  
Davies-Bouldin = mean(R_i)
```

**Interpretation:**
- Measures cluster cohesion vs. separation
- Lower = better-defined clusters with less overlap
- 0 = Perfect (clusters at different locations with no overlap)

**Results:**
- K-Means: 0.4534 → Well-separated clusters
- DBSCAN: 0.1072 → Highly separated clusters

---

### Calinski-Harabasz Index (0 to infinity, higher is better)

**Formula:**
```
Calinski-Harabasz = (Between-cluster variance) / (Within-cluster variance)
```

**Interpretation:**
- Ratio of between-cluster to within-cluster dispersion
- Higher = clusters are far apart AND compact
- More useful for comparing different k values

**Results:**
- K-Means: 460,933 (very high)
- DBSCAN: 12,072 (lower, but uses different cluster structure)

---

### Adjusted Rand Index (-1 to 1, higher is better)

**Formula:**
```
Measures agreement between two cluster assignments
Adjusted for chance agreement
```

**Interpretation:**
- **1** = Perfect agreement between clusterings
- **0** = Random agreement (by chance)
- **Negative** = Worse than random

**Results:**
- K-Means: 0.0001 (almost random)
- DBSCAN: 0.0002 (almost random)
- **Meaning:** Neither algorithm recovers predefined clusters

---

## 🎓 Unsupervised Learning Insights

### When Does Unsupervised Clustering Work?

**Success Case:**
- Natural groupings exist in the data
- Features are informative about group membership
- Groups have similar internal density

**Challenge Case (This Project):**
- Predefined clusters based on exact product matching
- Text similarity alone insufficient to recover exact matches
- Different clustering structure discovered

---

### Unsupervised vs. Supervised Learning

| Aspect | Supervised | Unsupervised |
|--------|-----------|-------------|
| **Data** | Features + Labels | Features only |
| **Goal** | Predict labels | Find patterns |
| **Evaluation** | Accuracy, recall, F1 | Silhouette, Davies-Bouldin |
| **Ground Truth** | Needed for validation | Optional (no labels) |
| **Result** | Model for prediction | Insights into structure |

---

## 🚀 How to Run the Clustering Analysis

### Quick Demo (Recommended Parameters)

```powershell
@("5", "1", "0.5", "3", "no") | python Clustering_Analysis.py
```

**Parameters:**
- 5 clusters (K-Means)
- Features: Product Title + Merchant ID
- Epsilon: 0.5 (DBSCAN)
- Algorithm: Both (3)
- Run again: No

**Expected Output:**
```
K-Means Results:
  Silhouette: 0.6965 (good)
  5 balanced clusters

DBSCAN Results:
  Silhouette: 0.9226 (excellent)
  1,281 fine-grained clusters
  39.53% noise points
```

### Exploring Different Parameters

**K-Means with Different k:**
```powershell
# Try 10 clusters instead of 5
@("10", "1", "0.5", "1", "no") | python Clustering_Analysis.py
```

**DBSCAN with Different Epsilon:**
```powershell
# Try larger epsilon (1.5) for fewer, larger clusters
@("5", "1", "1.5", "2", "no") | python Clustering_Analysis.py
```

**Try Different Features:**
```powershell
# Use text only (no merchant ID)
@("5", "2", "0.5", "3", "no") | python Clustering_Analysis.py

# Use merchant ID only
@("5", "3", "0.5", "3", "no") | python Clustering_Analysis.py
```

---

## 📊 Output Visualizations

### kmeans_analysis.png

**Left Panel:** Cluster size distribution
- Shows how K-Means divides the 35,311 products into 5 clusters
- Percentages for each cluster shown on bars

**Right Panel:** Quality metrics
- Silhouette, Calinski-Harabasz, Adjusted Rand scores
- Green/blue bars indicate metric values

### dbscan_analysis.png

**Left Panel:** Cluster distribution
- Shows all 1,281 clusters (many small bars!)
- Red portion shows noise points

**Right Panel:** Configuration summary
- Epsilon value, results summary
- Highlights outlier detection

---

## 💡 Key Learnings

### 1. Unsupervised Learning Discovers Different Structure

- Unsupervised clustering found patterns based on text & merchant features
- Predefined clusters based on exact product matching
- Different objectives → Different cluster structures
- Both valid, just different!

### 2. Evaluation Metrics Matter

- Quality metrics (Silhouette, Davies-Bouldin) say both algorithms work well
- Adjusted Rand Index (comparing to ground truth) shows disagreement
- Need BOTH perspectives for full understanding

### 3. Algorithm Choice Affects Results

- K-Means: Forced to use exactly k clusters
- DBSCAN: Discovers natural granularity
- Same data, different insights

### 4. Parameters Are Critical

- K-Means k affects number of clusters
- DBSCAN epsilon affects cluster granularity
- Small parameter changes → Big output changes

### 5. Unsupervised ≠ Automatic Ground Truth

- No labels doesn't mean no structure
- Structure exists but may not match expectations
- Must interpret results in domain context

---

## 🔄 Comparing All Three Projects

| Project | Type | Method | Accuracy | Goal |
|---------|------|--------|----------|------|
| **Naive Bayes** | Supervised | Probabilistic | 81.14% | Predict category from title |
| **ID3 Tree** | Supervised | Tree-based | 57.27% | Predict category from title |
| **K-Means** | Unsupervised | Partitioning | Silhouette: 0.70 | Find natural groupings |
| **DBSCAN** | Unsupervised | Density-based | Silhouette: 0.92 | Find fine-grained clusters |

**Key Insight:**
- Supervised (Classification): 81% accuracy achieved
- Unsupervised (Clustering): High-quality clusters found, but different structure

---

## 🎯 Conclusions

### Q: Can we cluster products without predefined categories?

**A: Yes, with caveats:**

✅ **Success:**
- Both K-Means and DBSCAN produce high-quality clusters (Silhouette > 0.6)
- Natural groupings discovered based on text similarity
- Outliers identified (DBSCAN detects 39.53% as noise)

❌ **Challenge:**
- Discovered clusters don't match predefined Cluster IDs
- Ground truth likely based on different criteria
- Unsupervised learning finds DATA structure, not labels

### Recommendation

**Use K-Means when:**
- You need exactly k groups
- You want interpretable segments
- Fast computation matters

**Use DBSCAN when:**
- You want to explore natural granularity
- You need outlier detection
- Cluster sizes can vary

**Use Supervised Classification when:**
- You have labeled training data
- You need to predict categories
- Accuracy is paramount (81% vs. unlabeled clustering)

---

## 📚 References

**Clustering Concepts:**
- K-Means: MacQueen (1967)
- DBSCAN: Ester et al. (1996)
- Silhouette Score: Rousseeuw (1987)
- Davies-Bouldin Index: Davies & Bouldin (1979)

**Libraries Used:**
- scikit-learn: Clustering algorithms
- scipy: Hierarchical clustering
- matplotlib/seaborn: Visualizations

---

**Project Status:** ✅ Complete  
**Clustering Script:** `Clustering_Analysis.py`  
**Visualizations:** `kmeans_analysis.png`, `dbscan_analysis.png`  

