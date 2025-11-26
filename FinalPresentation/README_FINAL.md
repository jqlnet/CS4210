# PriceRunner Product Classification - Final Presentation

## 📊 Project Summary

This project implements and compares two supervised machine learning algorithms for product category classification:

- **🟢 Naive Bayes (MultinomialNB)**: 81.14% accuracy ✅ RECOMMENDED
- **🔴 ID3 Decision Tree**: 57.27% accuracy ❌ (for reference/learning)

**Dataset:** 35,311 e-commerce products across 10 product categories

---

## 🚀 Quick Start

### Running Naive Bayes Classifier

```powershell
cd "c:\Users\Toni\vscode\Python\4210\FinalPresentation"
python NaiveBayes_Analysis.py
```

**Expected Results:**
- Test Accuracy: ~81.14%
- Cross-Validation Mean: ~81.71%
- CV Consistency: Excellent (std dev 0.0066)

**Interactive Prompts:**
```
1. Enter training set percentage (default 80): 80
2. Enter number of CV folds (default 5): 5
3. Enter alpha smoothing parameter (default 1.0): 1.0
4. Feature selection (1=Title+Merchant, 2=Title only): 1
5. Run again? (yes/no): no
```

### Running ID3 Decision Tree (Comparison)

```powershell
python ClassificationID3_Presentation.py
```

**Expected Results:**
- Test Accuracy: ~57.27%
- Cross-Validation Mean: ~55.94%
- CV Consistency: Poor (std dev 0.0615)

**Interactive Prompts:**
```
1. Enter training set percentage (default 80): 80
2. Enter max_depth (1-20, default 5): 5
3. Enter number of CV folds (default 5): 10
4. Enter min_samples_split (default 2): 5
5. Feature selection (1=Title+Merchant, 2=Title only): 1
6. Run again? (yes/no): no
```

---

## 📁 Project Structure

```
FinalPresentation/
├── NaiveBayes_Analysis.py              ← PRIMARY MODEL ✅
├── ClassificationID3_Presentation.py   ← COMPARISON MODEL
├── pricerunner_aggregate.csv           ← INPUT DATASET
│
├── PROJECT_SUMMARY.md                  ← Project overview
├── COMPARISON_RESULTS.md               ← Detailed analysis
├── README_PRESENTATION.md              ← Original setup notes
│
└── Output Visualizations/
    ├── naivebayes_confusion_matrix.png
    ├── naivebayes_cv_scores.png
    ├── confusion_matrix_id3.png
    └── decision_tree_plot.png
```

---

## 🎯 Key Results

### Performance Comparison

| Metric | Naive Bayes | ID3 Tree | Winner |
|--------|-------------|----------|--------|
| Test Accuracy | **81.14%** | 57.27% | 🟢 Naive Bayes |
| CV Mean | **81.71%** | 55.94% | 🟢 Naive Bayes |
| CV Std Dev | **0.0066** | 0.0615 | 🟢 Naive Bayes (9× better) |
| Generalization | ✅ Good | ❌ Poor | 🟢 Naive Bayes |

**Conclusion:** Naive Bayes dominates with 23.87% accuracy improvement and far superior generalization.

---

## 📈 Naive Bayes Performance by Class

| Class | Precision | Recall | F1-Score | Assessment |
|-------|-----------|--------|----------|-----------|
| 0 | 0.99 | **1.00** | **1.00** | ✅ Perfect |
| 1 | 0.82 | 0.85 | 0.83 | ✅ Excellent |
| 2 | 0.87 | 0.91 | 0.89 | ✅ Excellent |
| 3 | 0.94 | 0.50 | 0.65 | ⚠️ Low Recall |
| 4 | 0.69 | 0.76 | 0.72 | ✅ Good |
| 5 | 0.45 | 0.77 | 0.57 | ⚠️ Low Precision |
| 6 | **0.98** | 0.77 | 0.86 | ✅ Excellent |
| 7 | 0.97 | 0.72 | 0.82 | ✅ Excellent |
| 8 | 0.95 | 0.87 | **0.91** | ✅ Excellent |
| 9 | 0.98 | 0.88 | **0.92** | ✅ Excellent |

**Note:** Classes 3 and 5 show systematic confusion, likely due to genuine category overlap in the data.

---

## 🔍 Technical Details

### Feature Engineering

**Input Features:**
1. **Product Title (100 features)**
   - Processed using CountVectorizer
   - Top 100 most frequent words (excluding stopwords)
   - Bag-of-words representation (word counts)

2. **Merchant ID (1 feature)**
   - Categorical encoding (LabelEncoder)
   - Represents product source/seller

**Total: 101 features**

### Preprocessing Pipeline

```python
1. Load CSV data (35,311 samples × 7 columns)
2. Extract Product Title → CountVectorizer (100 text features)
3. Extract Merchant ID → LabelEncoder (1 categorical feature)
4. Combine with scipy.sparse.hstack
5. Convert to dense array (required for Decision Trees)
6. Create X (101 features) and y (10 class labels)
```

### Model Configuration

**Naive Bayes (MultinomialNB):**
```python
CountVectorizer(max_features=100, lowercase=True, stop_words='english')
MultinomialNB(alpha=1.0)  # Laplace smoothing
```

**ID3 Decision Tree:**
```python
DecisionTreeClassifier(
    criterion='entropy',      # ID3 splitting criterion
    max_depth=5,             # Limit tree complexity
    min_samples_split=5      # Prevent overfitting
)
```

### Evaluation Methodology

**Train/Test Split:**
- 80% training, 20% testing
- Random split (random_state=42 for reproducibility)

**Cross-Validation:**
- 5-fold K-fold cross-validation
- Provides robust accuracy estimate
- Lower std dev = more reliable model

**Metrics:**
- Accuracy: Overall correctness
- Precision: Avoid false positives
- Recall: Avoid false negatives
- F1-Score: Harmonic mean (balanced metric)
- Confusion Matrix: Detailed misclassification patterns

---

## 💡 Why Naive Bayes Wins

### 1. **Text-Optimized Algorithm**
- MultinomialNB designed specifically for bag-of-words features
- Treats word counts as discrete distributions
- Natural probability framework for text

### 2. **Excellent Generalization**
- CV std dev 0.0066 (extremely consistent)
- Test accuracy (81.14%) ≈ CV mean (81.71%)
- Indicates genuine learning, not overfitting

### 3. **Robust Class Handling**
- All 10 classes learned effectively
- Even minority classes achieve reasonable performance
- No complete class failures

### 4. **Simple Yet Effective**
- Fewer parameters to tune
- Linear decision boundaries
- Fast training and inference

---

## 🐛 Why ID3 Tree Struggles

### 1. **High-Dimensional Feature Space**
- 101 dimensions is challenging for shallow trees
- max_depth=5 insufficient for learning all classes
- Tree gets stuck partitioning high-dimensional space

### 2. **Poor Generalization**
- CV std dev 0.0615 (9× worse than Naive Bayes)
- Test accuracy > CV mean (indicates overfitting)
- Each fold produces vastly different tree

### 3. **Catastrophic Class Failures**
- Classes 1, 2, 6 achieve 0% recall (completely missed)
- Tree doesn't learn patterns for these classes
- Falls back to default predictions

### 4. **Feature Type Mismatch**
- Decision trees prefer dense, structured features
- Text features are sparse, high-dimensional
- Poor fit for tree-based splitting

---

## 🎓 Educational Value

This project demonstrates:

1. **Algorithm Selection is Critical**
   - Same features, different algorithms = 24% accuracy gap
   - Must consider feature type and problem characteristics

2. **Cross-Validation Reveals Truth**
   - Single accuracy can be misleading
   - CV std dev shows model stability
   - Generalization matters more than single test score

3. **Feature Engineering Foundations**
   - CountVectorizer parameters matter (max_features, stop_words)
   - Text preprocessing significant impact
   - Sparse vs. dense representations

4. **Error Analysis Guides Improvement**
   - Systematic confusion (NB): Data ambiguity
   - Complete failure (ID3): Architecture mismatch
   - Different solutions for different error types

5. **Simple Often Beats Complex**
   - Naive Bayes (simple) > ID3 (complex) for this task
   - Don't use complex models just because they exist
   - Let data guide algorithm choice

---

## 🔧 Customization

### Adjusting Naive Bayes

**Change Alpha Smoothing:**
- Lower alpha (0.1): More aggressive (less smoothing)
- Higher alpha (10.0): More conservative (more smoothing)
- Default (1.0): Laplace smoothing, standard choice

**Change Feature Count:**
Edit line in `NaiveBayes_Analysis.py`:
```python
vectorizer = CountVectorizer(max_features=100, ...)  # Change 100 to desired count
```

**Use Only Product Title (exclude Merchant ID):**
- Select option 2 when prompted
- Compares text-only classification

### Adjusting Decision Tree

**Increase Tree Depth:**
Change max_depth (default 5):
```python
max_depth = 10  # Deeper tree, more complex
```
- Risk: Overfitting
- Benefit: Learn more detailed patterns

**Adjust Min Samples Split:**
Change min_samples_split (default 5):
```python
min_samples_split = 2  # More aggressive splitting
```
- Lower values = smaller leaves
- Higher values = simpler tree

---

## 📊 Output Files

### Visualizations Generated

**For Naive Bayes:**
1. `naivebayes_confusion_matrix.png` - 10×10 matrix showing prediction accuracy per class
2. `naivebayes_cv_scores.png` - Line graph of cross-validation performance across 5 folds

**For ID3 Decision Tree:**
1. `confusion_matrix_id3.png` - 10×10 matrix showing tree prediction accuracy

### Interpretation Guide

**Confusion Matrix:**
- Diagonal values: Correct predictions
- Off-diagonal: Misclassifications
- Darker blue = higher values
- Shows which classes are confused with each other

**CV Scores Graph:**
- X-axis: Fold number (1-5)
- Y-axis: Accuracy (0-1)
- Horizontal line: Mean accuracy
- Low variation = consistent model
- High variation = unstable model

---

## ✅ Verification Checklist

Before presenting, verify:

- [ ] Both Python scripts run without errors
- [ ] Naive Bayes achieves ~81% accuracy
- [ ] ID3 Tree achieves ~57% accuracy
- [ ] Confusion matrix images generated correctly
- [ ] CV scores image shows Naive Bayes consistency
- [ ] All class distributions display correctly
- [ ] Cross-validation std dev reported accurately

---

## 🎬 Presentation Flow

1. **Problem Statement** (1 min)
   - Classify products into categories
   - Why it matters for e-commerce

2. **Dataset Overview** (1 min)
   - 35,311 products, 10 balanced categories
   - Features: Product Title + Merchant ID

3. **Feature Engineering** (2 min)
   - CountVectorizer explanation
   - Why 100 features?
   - Stop words removal

4. **Algorithm Comparison** (3 min)
   - Naive Bayes concept
   - ID3 Decision Tree concept
   - Show side-by-side results table

5. **Results Analysis** (3 min)
   - Performance comparison (81% vs 57%)
   - Confusion matrices: What's being confused?
   - Cross-validation std dev importance

6. **Key Insights** (2 min)
   - Text features need probabilistic models
   - Generalization matters more than single metric
   - Algorithm-feature matching critical

7. **Conclusion** (1 min)
   - Naive Bayes recommended for production
   - Why decision trees failed
   - Lessons learned

**Total Duration:** ~13 minutes

---

## 📚 References

**Scikit-Learn Documentation:**
- MultinomialNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
- DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

**ML Concepts:**
- Naive Bayes: Probabilistic classifier using Bayes' theorem
- Decision Trees: Hierarchical splitting using information gain (entropy)
- Cross-Validation: K-fold validation for robust model assessment
- Confusion Matrix: Detailed breakdown of prediction accuracy per class

---

## 🤝 Support

**Common Issues:**

Q: "Script asks for input but doesn't respond to my typing"
A: Use PowerShell pipe: `@("80", "5", "1.0", "1", "no") | python script.py`

Q: "Why does Naive Bayes achieve higher accuracy?"
A: Text features are best suited for probabilistic bag-of-words models, not tree-based approaches.

Q: "Can I improve the model further?"
A: Yes - add domain-specific features, use ensemble methods, explore transformer models.

Q: "How do I deploy this?"
A: Save the trained model using joblib, create Flask API wrapper, containerize with Docker.

---

## 📝 License & Attribution

This project uses:
- PriceRunner dataset (for educational purposes)
- Scikit-learn machine learning library
- Python ecosystem (pandas, numpy, matplotlib, seaborn)

**Created:** Final Presentation Project
**Status:** ✅ Complete and ready for presentation

---

**Happy Presenting! 🎉**

For questions or customizations, refer to the detailed analysis in `COMPARISON_RESULTS.md` and `PROJECT_SUMMARY.md`.

