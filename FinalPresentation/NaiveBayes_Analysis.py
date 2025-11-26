"""
Naive Bayes Classifier for PriceRunner Dataset
Comparison with ID3 Decision Tree and Logistic Regression

Author: Data Analysis Tool
Purpose: Demonstrate Naive Bayes performance on multi-class classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


def get_user_input():
    """Get configuration parameters from user"""
    print("\n" + "="*70)
    print("NAIVE BAYES CLASSIFIER - CONFIGURATION")
    print("="*70)
    
    # Train-test split
    print("\n[TRAINING/TEST SPLIT]")
    print("  Suggested: 80% (standard practice)")
    print("  Range: 1-99 (must be between 0 and 100)")
    train_input = input("  Enter training set percentage (or press Enter for 80): ").strip()
    train_size = 80 if train_input == "" else int(train_input)
    print(f"  -> Selected: {train_size}% training, {100-train_size}% testing")
    
    # CV folds
    print("\n[CROSS-VALIDATION FOLDS]")
    print("  Suggested: 5 (standard k-fold)")
    print("  Range: 2-10 (more folds = more robust)")
    cv_input = input("  Enter number of CV folds (or press Enter for 5): ").strip()
    cv_folds = 5 if cv_input == "" else int(cv_input)
    print(f"  -> Selected: {cv_folds}-fold cross-validation")
    
    # Alpha (smoothing parameter)
    print("\n[ALPHA (SMOOTHING)]")
    print("  Suggested: 1.0 (Laplace smoothing, handles unseen features)")
    print("  Range: 0.0-5.0 (higher = more smoothing)")
    alpha_input = input("  Enter alpha (or press Enter for 1.0): ").strip()
    alpha = 1.0 if alpha_input == "" else float(alpha_input)
    print(f"  -> Selected: {alpha}")
    
    # Feature selection
    print("\n[FEATURE SELECTION]")
    print("  Suggested: Option 1 (all features)")
    print("  Available options:")
    print("    1. All features (Product Title, Merchant ID, Cluster ID)")
    print("    2. Core features only (Product Title, Cluster ID)")
    print("    3. Minimal features (Product Title only)")
    feature_input = input("  Select option (or press Enter for 1): ").strip()
    feature_choice = 1 if feature_input == "" else int(feature_input)
    print(f"  -> Selected: Option {feature_choice}")
    
    return train_size, cv_folds, alpha, feature_choice


def load_and_prepare_data(feature_choice):
    """Load and encode data"""
    print("\n" + "="*70)
    print("DATA LOADING AND PREPARATION")
    print("="*70)
    
    # Load dataset
    df = pd.read_csv('pricerunner_aggregate.csv')
    print(f"\n[OK] Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
    
    print(f"\nData Verification:")
    print(f"  - Missing values: {df.isnull().sum().sum()} total")
    print(f"  - Columns: {list(df.columns)}")
    
    # Select features based on user choice
    if feature_choice == 1:
        features = ['Product Title', ' Merchant ID', ' Cluster ID']
    elif feature_choice == 2:
        features = ['Product Title', ' Cluster ID']
    else:  # feature_choice == 3
        features = ['Product Title']
    
    X = df[features].copy()
    y = df[' Category Label'].copy()
    
    # Encode categorical features
    le_dict = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    print(f"[OK] Categorical features encoded")
    print(f"[OK] Using {len(features)} feature(s): {', '.join(features)}")
    
    # Encode target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    return X, y, features


def display_data_summary(X, y, features):
    """Display summary statistics"""
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    print(f"\nFeatures used ({len(features)}):")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")
    
    print(f"\nTarget variable: Category Label (10 classes)")
    print(f"Total samples: {len(X)}")
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples")


def train_and_evaluate(X, y, train_size, cv_folds, alpha):
    """Train model and evaluate performance"""
    print("\n" + "="*70)
    print("MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    # Train-test split
    train_pct = train_size / 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_pct, random_state=42, stratify=y
    )
    
    print(f"\nTrain-Test Split ({train_size}%/{100-train_size}%):")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    
    # Model configuration
    print(f"\nModel Configuration:")
    print(f"  - Type: Categorical Naive Bayes")
    print(f"  - Alpha (smoothing): {alpha}")
    print(f"  - Assumption: Features are conditionally independent given class")
    
    # Train model
    model = CategoricalNB(alpha=alpha)
    model.fit(X_train, y_train)
    print(f"[OK] Model trained successfully")
    
    # Cross-validation
    print(f"\nCross-Validation ({cv_folds}-Fold):")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    print(f"  - Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"  - Mean accuracy: {cv_scores.mean():.4f}")
    print(f"  - Std deviation: {cv_scores.std():.4f}")
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Results:")
    print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Generate visualizations
    print(f"\n[OK] Generating visualizations...")
    
    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Naive Bayes - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('naivebayes_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] naivebayes_confusion_matrix.png saved")
    
    # Cross-validation scores visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, cv_folds + 1), cv_scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
    plt.axhline(y=test_accuracy, color='g', linestyle='--', label=f'Test: {test_accuracy:.4f}')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy Score')
    plt.title('Naive Bayes - Cross-Validation Fold Scores', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('naivebayes_cv_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] naivebayes_cv_scores.png saved")
    
    return model, y_test, y_pred, test_accuracy, cv_scores


def display_detailed_results(y_test, y_pred, test_accuracy):
    """Display detailed classification metrics"""
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print(f"\n[METRIC EXPLANATIONS]")
    print(f"  Precision: Of predictions I made for class X, how many were correct?")
    print(f"  Recall: Of actual class X samples, how many did I correctly identify?")
    print(f"  F1-Score: Harmonic mean of precision and recall (balanced metric)")
    print(f"  Support: Number of actual samples for each class in test set")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\n[INTERPRETATION GUIDE]")
    print(f"  High Precision (>0.90): Avoid false positives (accurate when predicting)")
    print(f"  High Recall (>0.90): Avoid false negatives (catches most actual samples)")
    print(f"  High F1-Score (>0.90): Good balance between precision and recall")
    print(f"  Confusion Matrix diagonal: Correct predictions")
    print(f"  Confusion Matrix off-diagonal: Misclassifications")


def display_summary(test_accuracy, cv_scores):
    """Display analysis summary"""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nPerformance Summary:")
    print(f"  - Test accuracy: {test_accuracy:.4f}")
    print(f"  - Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"  - CV std deviation: {cv_scores.std():.4f}")
    
    # Comparison note
    gap = test_accuracy - cv_scores.mean()
    if gap > 0.05:
        print(f"\n[NOTE] Test accuracy {gap:.4f} higher than CV mean (possible overfitting)")
    elif gap < -0.05:
        print(f"\n[NOTE] Test accuracy {abs(gap):.4f} lower than CV mean (possible underestimation)")
    else:
        print(f"\n[OK] Model shows good generalization (test ~= CV)")
    
    print(f"\n" + "="*70)
    print(f"Output files saved:")
    print(f"  - naivebayes_confusion_matrix.png")
    print(f"  - naivebayes_cv_scores.png")
    print("="*70)


def main():
    """Main execution loop"""
    print("\n" + "="*70)
    print("        NAIVE BAYES CLASSIFIER - PRICERUNNER DATASET")
    print("           Classification Analysis and Evaluation Tool")
    print("="*70)
    
    while True:
        # Get user input
        train_size, cv_folds, alpha, feature_choice = get_user_input()
        
        # Load and prepare data
        X, y, features = load_and_prepare_data(feature_choice)
        
        # Display summary
        display_data_summary(X, y, features)
        
        # Train and evaluate
        model, y_test, y_pred, test_accuracy, cv_scores = train_and_evaluate(
            X, y, train_size, cv_folds, alpha
        )
        
        # Display results
        display_detailed_results(y_test, y_pred, test_accuracy)
        display_summary(test_accuracy, cv_scores)
        
        # Ask for another run
        print("\n" + "-"*70 + "\n")
        again = input("Would you like to run with different parameters? (yes/no): ").strip().lower()
        if again != 'yes':
            print("\n[OK] Analysis complete. Goodbye!")
            break


if __name__ == "__main__":
    main()
