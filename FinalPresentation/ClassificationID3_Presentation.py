# ============================================================
# ID3 DECISION TREE CLASSIFIER - PRICERUNNER AGGREGATE for Final Presentation 
# ============================================================

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def get_user_input():
    """Collect user configuration for model training"""
    print("\n" + "=" * 70)
    print("ID3 DECISION TREE CLASSIFIER - CONFIGURATION")
    print("=" * 70)
    
    # Training/Test split
    print("\n[TRAINING/TEST SPLIT]")
    print("  Suggested: 80% (standard practice)")
    print("  Range: 1-99 (must be between 0 and 100)")
    while True:
        try:
            user_input = input("  Enter training set percentage (or press Enter for 80): ").strip()
            if user_input == "":
                train_size = 80
            else:
                train_size = float(user_input)
            if 0 < train_size < 100:
                test_size = (100 - train_size) / 100
                train_size = train_size / 100
                print(f"  -> Selected: {train_size*100:.0f}% training, {test_size*100:.0f}% testing")
                break
            else:
                print("  ERROR: Please enter a value between 0 and 100")
        except ValueError:
            print("  ERROR: Please enter a valid number")
    
    # Max depth
    print("\n[MAX TREE DEPTH]")
    print("  Suggested: 5 (industry standard, good generalization)")
    print("  Range: 1-20 (higher = more complex tree)")
    while True:
        try:
            user_input = input("  Enter max_depth (or press Enter for 5): ").strip()
            if user_input == "":
                max_depth = 5
            else:
                max_depth = int(user_input)
            if 1 <= max_depth <= 20:
                print(f"  -> Selected: depth {max_depth}")
                break
            else:
                print("  ERROR: Please enter a value between 1 and 20")
        except ValueError:
            print("  ERROR: Please enter a valid integer")
    
    # Cross-validation folds
    print("\n[CROSS-VALIDATION FOLDS]")
    print("  Suggested: 5 (standard k-fold)")
    print("  Range: 2-10 (more folds = more robust)")
    while True:
        try:
            user_input = input("  Enter number of CV folds (or press Enter for 5): ").strip()
            if user_input == "":
                cv_folds = 5
            else:
                cv_folds = int(user_input)
            if 2 <= cv_folds <= 10:
                print(f"  -> Selected: {cv_folds}-fold cross-validation")
                break
            else:
                print("  ERROR: Please enter a value between 2 and 10")
        except ValueError:
            print("  ERROR: Please enter a valid integer")
    
    # Min samples split
    print("\n[MIN SAMPLES SPLIT]")
    print("  Suggested: 2 (allows fine-grained splitting)")
    print("  Range: 1-20 (higher = simpler tree, less overfitting)")
    while True:
        try:
            user_input = input("  Enter min_samples_split (or press Enter for 2): ").strip()
            if user_input == "":
                min_samples = 2
            else:
                min_samples = int(user_input)
            if 1 <= min_samples <= 20:
                print(f"  -> Selected: {min_samples}")
                break
            else:
                print("  ERROR: Please enter a value between 1 and 20")
        except ValueError:
            print("  ERROR: Please enter a valid integer")
    
    # Feature selection
    print("\n[FEATURE SELECTION]")
    print("  Suggested: Option 1 (all features)")
    print("  Available options:")
    print("    1. All features (Product Title, Merchant ID, Cluster ID)")
    print("    2. Core features only (Product Title, Cluster ID)")
    print("    3. Custom selection")
    
    feature_choice = input("  Select option (or press Enter for 1): ").strip() or "1"
    if feature_choice == "1":
        print("  -> Selected: All 3 features")
    elif feature_choice == "2":
        print("  -> Selected: Core features (2)")
    else:
        print("  -> Selected: Default features")
    
    return {
        'train_size': train_size,
        'test_size': test_size,
        'max_depth': max_depth,
        'cv_folds': cv_folds,
        'min_samples_split': min_samples,
        'feature_choice': feature_choice
    }

def load_and_prepare_data(config):
    """Load data and prepare features based on user config"""
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
    print(f"  - Columns: {df.columns.tolist()}")
    
    # Encode categorical columns
    df['Product Title'] = df['Product Title'].astype('category').cat.codes
    df[' Cluster Label'] = df[' Cluster Label'].astype('category').cat.codes
    df[' Category Label'] = df[' Category Label'].astype('category').cat.codes
    
    print("\n[OK] Categorical features encoded")
    
    # Select features based on user choice
    if config['feature_choice'] == '1':
        features = ['Product Title', ' Merchant ID', ' Cluster ID']
        print("[OK] Using all 3 features (Product Title, Merchant ID, Cluster ID)")
    elif config['feature_choice'] == '2':
        features = ['Product Title', ' Cluster ID']
        print("[OK] Using core features (Product Title, Cluster ID)")
    else:
        features = ['Product Title', ' Merchant ID', ' Cluster ID']
        print("[OK] Using default features (all 3)")
    
    target = ' Category Label'
    
    X = df[features]
    y = df[target]
    
    return X, y, features, target, df

def display_data_summary(X, y, features):
    """Display summary statistics about the data"""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"\nFeatures used ({len(features)}):")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")
    
    print(f"\nTarget variable: Category Label (10 classes)")
    print(f"Total samples: {len(y):,}")
    print(f"\nClass distribution:")
    print(y.value_counts().sort_index())

def train_and_evaluate(X, y, config):
    """Train model and perform comprehensive evaluation"""
    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=config['train_size'],
        random_state=42
    )
    
    print(f"\nTrain-Test Split ({config['train_size']:.0%}/{config['test_size']:.0%}):")
    print(f"  - Training samples: {len(X_train):,}")
    print(f"  - Test samples: {len(X_test):,}")
    
    # Initialize model
    clf = tree.DecisionTreeClassifier(
        criterion='entropy',
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        random_state=42
    )
    
    print(f"\nModel Configuration:")
    print(f"  - Criterion: entropy (ID3)")
    print(f"  - Max depth: {config['max_depth']}")
    print(f"  - Min samples split: {config['min_samples_split']}")
    
    # Train model
    clf.fit(X_train, y_train)
    print("\n[OK] Model trained successfully")
    
    # Cross-validation
    print(f"\nCross-Validation ({config['cv_folds']}-Fold):")
    cv_scores = cross_val_score(clf, X, y, cv=config['cv_folds'], scoring='accuracy')
    print(f"  - Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"  - Mean accuracy: {cv_scores.mean():.4f}")
    print(f"  - Std deviation: {cv_scores.std():.4f}")
    
    # Test predictions
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Results:")
    print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Generate visualizations
    print("\n[OK] Generating visualizations...")
    
    # Decision tree plot
    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, 
                   feature_names=X.columns.tolist(),
                   class_names=[f'Class {i}' for i in range(10)],
                   filled=True, 
                   rounded=True, 
                   fontsize=8)
    plt.title('ID3 Decision Tree Classifier', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('decision_tree_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] decision_tree_plot.png saved")
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] confusion_matrix_plot.png saved")
    
    return clf, X_train, X_test, y_train, y_test, y_pred, test_accuracy, cv_scores

def display_detailed_results(y_test, y_pred, test_accuracy):
    """Display comprehensive classification results"""
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print(f"\n[METRIC EXPLANATIONS]")
    print(f"  Precision: Of predictions I made for class X, how many were correct?")
    print(f"  Recall: Of actual class X samples, how many did I correctly identify?")
    print(f"  F1-Score: Harmonic mean of precision and recall (balanced metric)")
    print(f"  Support: Number of actual samples for each class in test set")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\n[INTERPRETATION GUIDE]")
    print(f"  High Precision (>0.90): Avoid false positives (accurate when predicting)")
    print(f"  High Recall (>0.90): Avoid false negatives (catches most actual samples)")
    print(f"  High F1-Score (>0.90): Good balance between precision and recall")
    print(f"  Confusion Matrix diagonal: Correct predictions")
    print(f"  Confusion Matrix off-diagonal: Misclassifications")

def display_summary(config, test_accuracy, cv_scores):
    """Display final summary"""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nConfiguration Summary:")
    print(f"  - Train/Test split: {config['train_size']:.0%}/{config['test_size']:.0%}")
    print(f"  - Max depth: {config['max_depth']}")
    print(f"  - Min samples split: {config['min_samples_split']}")
    print(f"  - Cross-validation folds: {config['cv_folds']}")
    
    print(f"\nPerformance Summary:")
    print(f"  - Test accuracy: {test_accuracy:.4f}")
    print(f"  - Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"  - CV std deviation: {cv_scores.std():.4f}")
    
    if abs(test_accuracy - cv_scores.mean()) < 0.1:
        print(f"\n[OK] Model shows good generalization (test ~= CV)")
    else:
        print(f"\n[WARNING] Model may be overfitting (test >> CV)")
    
    print("\n" + "=" * 70)
    print("Output files saved:")
    print("  - decision_tree_plot.png")
    print("  - confusion_matrix_plot.png")
    print("=" * 70 + "\n")

def main():
    """Main execution flow"""
    print("\n")
    print("=" * 70)
    print("  ID3 DECISION TREE CLASSIFIER - PRICERUNNER DATASET".center(70))
    print("  Classification Analysis and Evaluation Tool".center(70))
    print("=" * 70)
    
    run_again = True
    while run_again:
        # Get user configuration
        config = get_user_input()
        
        # Load and prepare data
        X, y, features, target, df = load_and_prepare_data(config)
        
        # Display data summary
        display_data_summary(X, y, features)
        
        # Train and evaluate
        clf, X_train, X_test, y_train, y_test, y_pred, test_accuracy, cv_scores = train_and_evaluate(X, y, config)
        
        # Display detailed results
        display_detailed_results(y_test, y_pred, test_accuracy)
        
        # Display summary
        display_summary(config, test_accuracy, cv_scores)
        
        # Ask if user wants to run again with different parameters
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
