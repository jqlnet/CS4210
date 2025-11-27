# Classification Metrics: Precision, Recall, and F1-Score Analysis

## Overview
This section provides detailed insights into how each model performs across precision, recall, and F1-score metrics for product category classification.

---

## Metric Definitions

**Precision:** Of all the products the model *predicted* as Category X, how many were actually correct?
- High precision = Few false positives (doesn't predict wrong categories)
- Formula: True Positives / (True Positives + False Positives)

**Recall:** Of all the actual Category X products in the test set, how many did the model correctly identify?
- High recall = Few false negatives (catches most actual products)
- Formula: True Positives / (True Positives + False Negatives)

**F1-Score:** Harmonic mean of precision and recall (balanced metric)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Best value: 1.0 (perfect), Worst value: 0.0

---

## Naive Bayes Classification Metrics

### Overall Performance
- **Test Accuracy:** 81.14%
- **Macro Average F1-Score:** 0.82
- **Weighted Average F1-Score:** 0.82

### Per-Class Performance

| Class | Category | Precision | Recall | F1-Score | Interpretation |
|-------|----------|-----------|--------|----------|-----------------|
| 0 | CPUs | 0.99 | 1.00 | 1.00 | ⭐ Perfect—catches all, no mistakes |
| 1 | Digital Cameras | 0.82 | 0.85 | 0.83 | ✓ Good—balanced performance |
| 2 | Dishwashers | 0.87 | 0.91 | 0.89 | ✓ Excellent—high recall, catches most |
| 3 | Freezers | 0.94 | 0.50 | 0.65 | ⚠️ High precision, but misses 50% |
| 4 | Fridge Freezers | 0.69 | 0.76 | 0.72 | ⚠️ Moderate—struggles with confusion |
| 5 | Fridges | 0.45 | 0.77 | 0.57 | ❌ Low precision—many false positives |
| 6 | Microwaves | 0.98 | 0.77 | 0.86 | ⚠️ High precision, misses 23% |
| 7 | Mobile Phones | 0.97 | 0.72 | 0.82 | ⚠️ High precision, misses 28% |
| 8 | TVs | 0.95 | 0.87 | 0.91 | ✓ Excellent—strong across both metrics |
| 9 | Washing Machines | 0.98 | 0.88 | 0.92 | ✓ Excellent—very reliable |

### Strengths
- **Classes 0, 8, 9:** Near-perfect to excellent performance (F1 > 0.90)
- **High Precision Categories:** Classes 0, 3, 6, 7, 9 (>0.94)—rarely makes false positive mistakes
- **Strong Recall:** Classes 0, 2, 5, 9 (>0.87)—catches most actual products

### Weaknesses
- **Class 5 (Fridges):** Only 45% precision—frequently misclassifies other appliances as fridges
- **Class 3 (Freezers):** Only 50% recall—misses half of actual freezers
- **Classes 4, 5, 7:** Moderate F1-scores (0.57-0.82)—indicates confusion with similar categories

---

## ID3 Decision Tree Classification Metrics

### Overall Performance
- **Test Accuracy:** 57.27%
- **Macro Average F1-Score:** 0.57
- **Weighted Average F1-Score:** 0.57

### Per-Class Performance

| Class | Category | Precision | Recall | F1-Score | Interpretation |
|-------|----------|-----------|--------|----------|-----------------|
| 0 | CPUs | 0.95 | 0.94 | 0.95 | ✓ Strong—one of the few good classes |
| 1 | Digital Cameras | 0.58 | 0.61 | 0.59 | ❌ Poor—low precision & recall |
| 2 | Dishwashers | 0.83 | 0.62 | 0.71 | ❌ Low recall—misses 38% |
| 3 | Freezers | 0.87 | 0.36 | 0.51 | ❌ Critical—only catches 36% |
| 4 | Fridge Freezers | 0.79 | 0.72 | 0.75 | ⚠️ Moderate |
| 5 | Fridges | 0.61 | 0.65 | 0.63 | ❌ Poor—struggles significantly |
| 6 | Microwaves | 0.96 | 0.77 | 0.86 | ✓ Decent precision, misses 23% |
| 7 | Mobile Phones | 0.92 | 0.72 | 0.81 | ⚠️ Moderate—misses 28% |
| 8 | TVs | 0.94 | 0.87 | 0.91 | ✓ Strong performer |
| 9 | Washing Machines | 0.87 | 0.88 | 0.87 | ✓ Consistent performance |

### Strengths
- **Classes 0, 8, 9:** Acceptable performance (F1 > 0.87)
- **High Precision:** Classes 0, 6, 8, 9 avoid false positives well
- **Balanced Classes:** 4, 7, 9 have relatively consistent precision/recall

### Weaknesses
- **Class 3 (Freezers):** Critical failure—only 36% recall, misses 64% of freezers
- **Class 1 (Digital Cameras):** 58-61% performance across all metrics—very unreliable
- **Class 5 (Fridges):** 61-65% performance—struggles to distinguish from related categories
- **Overall Trend:** Most categories have >20% recall loss vs. Naive Bayes

---

## Side-by-Side Comparison: Critical Classes

### Class 2 (Dishwashers)
| Metric | Naive Bayes | ID3 | Difference |
|--------|-------------|-----|-----------|
| Precision | 0.87 | 0.83 | NB +0.04 |
| Recall | **0.91** | **0.62** | **NB +29%** ⭐ |
| F1-Score | **0.89** | **0.71** | **NB +0.18** |

**Verdict:** Naive Bayes catches 29% more dishwashers—critical advantage.

### Class 3 (Freezers)
| Metric | Naive Bayes | ID3 | Difference |
|--------|-------------|-----|-----------|
| Precision | 0.94 | 0.87 | NB +0.07 |
| Recall | **0.50** | **0.36** | ID3 worse by 14% |
| F1-Score | **0.65** | **0.51** | **NB +0.14** |

**Verdict:** Both models struggle, but Naive Bayes is more reliable.

### Class 5 (Fridges)
| Metric | Naive Bayes | ID3 | Difference |
|--------|-------------|-----|-----------|
| Precision | 0.45 | 0.61 | ID3 +0.16 |
| Recall | **0.77** | **0.65** | **NB +0.12** |
| F1-Score | **0.57** | **0.63** | ID3 +0.06 |

**Verdict:** ID3 has higher precision (fewer false positives), but Naive Bayes catches more fridges.

---

## Key Insights

### 1. **Algorithm-Data Mismatch**
- **ID3:** Designed for low-dimensional categorical data; struggles with 100 text features
- **Naive Bayes:** Probabilistic model; naturally suited for text bag-of-words representation

### 2. **Recall Crisis in ID3**
- Freezers (Class 3): Only 36% recall
- Digital Cameras (Class 1): Only 61% recall  
- Dishwashers (Class 2): Only 62% recall
- **Why:** ID3 tends to overfit to dominant classes, missing minority classes

### 3. **Precision Concerns in Naive Bayes**
- Fridges (Class 5): Only 45% precision
- Fridge Freezers (Class 4): Only 69% precision
- **Why:** Text features for similar appliances overlap (e.g., "fridge" text in both Fridges and Fridge Freezers)

### 4. **Best Performing Classes**
- **Both Models Excel:** Classes 0 (CPUs), 8 (TVs), 9 (Washing Machines)
- **Why:** These categories have distinct text patterns; less confusion with others

### 5. **Tradeoff Pattern**
- Naive Bayes: **Better recall** (catches products), occasionally low precision (false alarms)
- ID3: **Better precision** (fewer false alarms), but low recall (misses products)

---

## Recommendations

### For Naive Bayes
✓ **Use when:** You want to capture as many products as possible (minimize false negatives)
- Good for: Search results, product recommendations, exploratory analysis
- ⚠️ Accept some false positives in Classes 4-5

### For ID3
✗ **Not recommended for this dataset**
- ID3's recall failures make it unreliable for production use
- Even if precision is acceptable, missing 28-64% of products is unacceptable

### Future Improvements
1. **Feature Engineering:** Create category-specific features (e.g., "door" count for fridges)
2. **Class Balancing:** Use class weights to address minority classes
3. **Ensemble Methods:** Combine Naive Bayes with other algorithms
4. **Hyperparameter Tuning:** (Naive Bayes only; ID3 fundamentally unsuited)

---

## Conclusion

**Naive Bayes is the clear winner** for product category classification on this dataset:
- 24% higher overall accuracy (81% vs 57%)
- Better recall across most classes
- More reliable predictions for business decisions
- Better suited for text data (bag-of-words features)

ID3's poor performance demonstrates that **algorithm selection matters more than tuning**. The fundamental mismatch between ID3 (designed for categorical data) and text features (100-dimensional sparse vectors) cannot be fixed through parameter adjustment.
