# Machine Learning Classification Comparison
## PriceRunner Product Dataset Analysis

**Date:** Final Presentation Project  
**Dataset:** 35,311 product samples across 10 categories  
**Preprocessing:** Text vectorization with CountVectorizer (100 features) + Merchant ID encoding (1 feature)  
**Total Features:** 101  

---

## Executive Summary

This analysis compares two supervised classification algorithms using identical preprocessing:

1. **Naive Bayes (MultinomialNB)** - Probabilistic text classifier
2. **ID3 Decision Tree** - Entropy-based tree classifier

Both models use the same feature engineering pipeline:
- **Input:** Product Title (via CountVectorizer) + Merchant ID
- **Target:** Product Category (10 classes)
- **Train/Test Split:** 80/20
- **Cross-Validation:** 5-fold K-fold CV

---

## Performance Comparison

### Overall Accuracy

| Metric | Naive Bayes | ID3 Decision Tree | Difference |
|--------|-------------|------------------|-----------|
| **Test Accuracy** | **81.14%** | 57.27% | +23.87% |
| **CV Mean** | **81.71%** | 55.94% | +25.77% |
| **CV Std Dev** | **0.0066** (excellent) | 0.0615 (poor) | -0.0549 |
| **Generalization** | ✅ Good (test ≈ CV) | ❌ Poor (test > CV) |

**Key Finding:** Naive Bayes significantly outperforms ID3 Decision Tree on this dataset with:
- 23.87% higher test accuracy
- Much better cross-validation consistency (std dev 0.0066 vs 0.0615)
- Better generalization (test accuracy close to CV mean)

---

## Detailed Classification Metrics

### Naive Bayes - Class-by-Class Performance

| Class | Precision | Recall | F1-Score | Support | Assessment |
|-------|-----------|--------|----------|---------|-----------|
| **0** | 0.99 | **1.00** | **1.00** | 773 | ✅ Excellent |
| 1 | 0.82 | 0.85 | 0.83 | 540 | Good |
| 2 | 0.87 | 0.91 | 0.89 | 685 | Good |
| 3 | 0.94 | 0.50 | 0.65 | 442 | ⚠️ Low Recall |
| 4 | 0.69 | 0.76 | 0.72 | 1,100 | Moderate |
| 5 | 0.45 | 0.77 | 0.57 | 717 | ⚠️ Low Precision |
| 6 | **0.98** | 0.77 | **0.86** | 468 | ✅ Excellent Precision |
| 7 | 0.97 | 0.72 | 0.82 | 816 | Good |
| 8 | 0.95 | 0.87 | **0.91** | 713 | ✅ Excellent |
| 9 | **0.98** | 0.88 | **0.92** | 809 | ✅ Excellent |

**Naive Bayes Summary:**
- **Weighted F1-Score:** 0.82 (very good balance)
- **Macro F1-Score:** 0.82 (stable across classes)
- **Strong Classes:** 0 (perfect recall), 6 (0.98 precision), 8/9 (f1 > 0.91)
- **Weak Classes:** 3 (50% recall), 5 (45% precision) - systematic confusion

---

### ID3 Decision Tree - Class-by-Class Performance

| Class | Precision | Recall | F1-Score | Support | Assessment |
|-------|-----------|--------|----------|---------|-----------|
| 0 | 1.00 | 0.67 | 0.80 | 771 | ⚠️ Low Recall |
| 1 | 0.00 | 0.00 | 0.00 | 542 | ❌ Fails Entirely |
| 2 | 0.00 | 0.00 | 0.00 | 662 | ❌ Fails Entirely |
| 3 | 0.98 | 0.78 | 0.87 | 439 | Good |
| 4 | 0.97 | 0.73 | 0.83 | 1,115 | Good |
| 5 | 0.24 | 0.71 | 0.35 | 728 | ❌ Poor Balance |
| 6 | 0.00 | 0.00 | 0.00 | 469 | ❌ Fails Entirely |
| 7 | 0.38 | 1.00 | 0.56 | 818 | ⚠️ Very Low Precision |
| 8 | 1.00 | 0.77 | 0.87 | 723 | Good |
| 9 | 1.00 | 0.61 | 0.76 | 796 | ⚠️ Low Recall |

**ID3 Decision Tree Summary:**
- **Weighted F1-Score:** 0.55 (poor balance)
- **Macro F1-Score:** 0.50 (unstable across classes)
- **Major Issues:**
  - Classes 1, 2, 6 completely missed (0% recall)
  - Class 7: 100% recall but only 38% precision (many false positives)
  - Class 5: 24% precision (unreliable predictions)

---

## Confusion Matrix Analysis

### Naive Bayes - Main Confusion Patterns

```
Primary Misclassifications:
- Class 4 → Class 5: 223 errors (20% of class 4 samples)
- Class 5 → Class 4: 133 errors (18.5% of class 5 samples)
- Class 1 → Class 5: 48 errors
- Class 2 → Class 5: 62 errors

Observation: Classes 4 and 5 have significant overlap
             (both likely electronics/hardware categories)
```

### ID3 Decision Tree - Main Issues

```
Complete Failures:
- Class 1: All 542 samples misclassified (mostly to class 5/7)
- Class 2: All 662 samples misclassified (mostly to class 5/7)
- Class 6: All 469 samples misclassified (mostly to class 7)

Observation: Tree fails to learn these classes entirely,
             relying on "default" predictions
```

---

## Algorithm Analysis

### Why Naive Bayes Performs Better

1. **Text-Friendly:** MultinomialNB designed for bag-of-words features
   - Word counts naturally suit Naive Bayes' probabilistic framework
   - Laplace smoothing (alpha=1.0) handles rare words well

2. **Generalization:** Simpler model, less prone to overfitting
   - Linear decision boundaries in probabilistic space
   - Works well with high-dimensional sparse features

3. **Stability:** Consistent performance across CV folds
   - Std dev 0.0066 indicates very robust learning
   - Test accuracy (81.14%) ≈ CV mean (81.71%)

4. **Class Handling:** Better with imbalanced classes
   - Learns probabilistic patterns for all 10 classes
   - Doesn't abandon difficult classes

### Why ID3 Decision Tree Struggles

1. **Tree Overfitting:** Shallow tree (max_depth=5) insufficient
   - Tree must split on text features (100+ choices per node)
   - Limited depth = coarse partitioning
   - Result: complete classes missed

2. **Feature Representation:** Sparse text features don't suit trees
   - Decision trees prefer categorical/numeric features
   - Text word counts are high-dimensional
   - Tree can't efficiently partition 100-dimensional space

3. **Poor Generalization:** Large CV std dev (0.0615)
   - Different folds produce vastly different trees
   - Test accuracy (57.27%) > CV mean (55.94%) = overfitting
   - Indicates unstable, unreliable model

4. **Class Collapse:** Missing entire classes
   - Not enough tree depth to learn rare class patterns
   - Min_samples_split=5 too restrictive at leaves
   - Results in "default" predictions for abandoned classes

---

## Configuration Parameters

### Naive Bayes (MultinomialNB)

```python
alpha = 1.0                    # Laplace smoothing
max_features = 100             # Text vocabulary size
stop_words = 'english'         # Remove common words
train_size = 0.80              # 80/20 split
cv_folds = 5                   # 5-fold cross-validation
```

**Why These Work:**
- alpha=1.0: Standard practice for text, avoids zero probabilities
- max_features=100: Good balance (reduces noise, keeps signal)
- stop_words removal: Improves feature quality for classification

---

### ID3 Decision Tree

```python
criterion = 'entropy'          # ID3 splitting criterion
max_depth = 5                  # Maximum tree depth
min_samples_split = 5          # Minimum samples to split
train_size = 0.80              # 80/20 split
cv_folds = 10                  # 10-fold cross-validation
```

**Why These Don't Optimize Performance:**
- max_depth=5: Too shallow for 101 features
  - Would need depth 10-15 to learn all classes
  - But increases overfitting significantly
- min_samples_split=5: Too restrictive at leaf level
  - Results in many pruned/abandoned class patterns

---

## Key Insights

### 1. **Text Features Need Probabilistic Models**
   - Bag-of-words → Naive Bayes ✅
   - Bag-of-words → Decision Trees ❌

### 2. **Class Imbalance Tolerance**
   - Naive Bayes: Handles imbalanced classes gracefully
   - ID3: Struggles with minority classes (especially at shallow depth)

### 3. **Generalization Matters**
   - Naive Bayes: Consistent (std dev 0.0066)
   - ID3: Unstable (std dev 0.0615 - 9x worse!)

### 4. **Systematic Errors vs. Complete Failure**
   - Naive Bayes: Systematic confusion (classes 4↔5)
     - Indicates real data challenge, not model failure
   - ID3: Complete failure for classes 1, 2, 6
     - Indicates model architecture mismatch

---

## Recommendations

### For This Dataset

**Use Naive Bayes** as the primary model:
- ✅ 81.14% accuracy (realistic, honest performance)
- ✅ Excellent generalization (CV std dev 0.0066)
- ✅ Interpretable probabilistic decisions
- ✅ Handles all 10 classes effectively

### To Improve Further

**Address Classes 3 and 5** (weak performers):
- These classes have genuine overlap in product characteristics
- Consider:
  1. Feature engineering: Add more discriminative text features
  2. Class rebalancing: Oversample minority classes in training
  3. Ensemble: Combine Naive Bayes with other classifiers

**Understand the Confusion**:
- Classes 4 and 5 systematically confused in Naive Bayes
- Likely both "Electronics" subcategories with overlapping terminology
- Suggests data labeling or category definition issue

---

## Conclusion

This analysis demonstrates that **algorithm selection matters** for specific problem types:

- **Text Classification:** Naive Bayes (81.14%) >> Decision Trees (57.27%)
- **Lesson:** Match algorithm to feature type
  - Text/Sparse features → Probabilistic models
  - Tabular/Dense features → Tree-based models

The Naive Bayes classifier is well-suited for this PriceRunner dataset and achieves strong, generalizable performance across all product categories.

---

**Generated:** Classification Analysis Project  
**Scripts:** `NaiveBayes_Analysis.py`, `ClassificationID3_Presentation.py`  
**Visualizations:** 
- `naivebayes_confusion_matrix.png`
- `naivebayes_cv_scores.png`
- `confusion_matrix_id3.png`
