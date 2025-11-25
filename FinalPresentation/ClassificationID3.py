# Comprehensive Decision Tree Evaluation Script for PriceRunner Dataset
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# the file to be opened
csv_file = 'pricerunner_aggregate.csv'

# Read data
df = pd.read_csv(csv_file)

# Data Verification
print("=" * 60)
print("DATA VERIFICATION")
print("=" * 60)
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst few rows:\n{df.head()}")

# Encode categorical columns as integers
df['Product Title'] = df['Product Title'].astype('category').cat.codes
df[' Merchant ID'] = df[' Merchant ID'].astype(int)
df[' Cluster Label'] = df[' Cluster Label'].astype('category').cat.codes
df[' Category Label'] = df[' Category Label'].astype('category').cat.codes

# REDUCED FEATURES (without Category ID and Cluster Label to test real performance)
# Only using: Product Title, Merchant ID, Cluster ID
features = ['Product Title', ' Merchant ID', ' Cluster ID']
target = ' Category Label'

print("\nFEATURES USED:")
print(f"  - Product Title (encoded)")
print(f"  - Merchant ID")
print(f"  - Cluster ID")
print(f"  (Removed: Cluster Label, Category ID to avoid overfitting leakage)\n")

X = df[features]
y = df[target]

# TEST: Remove 2 random features
import random
random.seed(42)
features_to_test = features.copy()
removed_features = random.sample(features_to_test, 2)
remaining_features = [f for f in features_to_test if f not in removed_features]

print("=" * 60)
print("FEATURE IMPORTANCE TEST - RANDOM REMOVAL")
print("=" * 60)
print(f"Original features: {features}")
print(f"Removed features: {removed_features}")
print(f"Remaining features: {remaining_features}\n")

# Split into training and testing sets (80/20 split, randomize for fairness)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the split
print("\n" + "=" * 60)
print("SPLIT VERIFICATION")
print("=" * 60)
print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Total: {len(X_train) + len(X_test)} samples")
print(f"\nTarget class distribution (overall):\n{y.value_counts().sort_index()}")
print(f"\nTarget class distribution (training set):\n{y_train.value_counts().sort_index()}")
print(f"\nTarget class distribution (test set):\n{y_test.value_counts().sort_index()}")

# Model with FULL FEATURES
print("\n" + "=" * 60)
print("MODEL 1: WITH ALL 3 FEATURES")
print("=" * 60)

# Initialize and train ID3 Decision Tree (using entropy)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train, y_train)

# CROSS-VALIDATION TEST (K-Fold with k=5)
print("=" * 60)
print("CROSS-VALIDATION RESULTS (5-Fold)")
print("=" * 60)
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Std Dev: {cv_scores.std():.4f}")
print(f"(If mean is similar to test accuracy, model is not overfitting)\n")

# Predictions
y_pred = clf.predict(X_test)

# Metrics and Reporting
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.3f}')

# Now test with REDUCED FEATURES (2 randomly removed)
print("\n" + "=" * 60)
print("MODEL 2: WITH REDUCED FEATURES (2 randomly removed)")
print("=" * 60)

X_reduced = df[remaining_features]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

clf_reduced = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf_reduced.fit(X_train_r, y_train_r)

y_pred_r = clf_reduced.predict(X_test_r)
accuracy_r = accuracy_score(y_test_r, y_pred_r)
print(f'Test Accuracy: {accuracy_r:.3f}')

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Accuracy with all 3 features: {accuracy:.3f}")
print(f"Accuracy with {remaining_features}: {accuracy_r:.3f}")
print(f"Difference: {abs(accuracy - accuracy_r):.3f}")
print(f"Impact of removed features {removed_features}: {(accuracy - accuracy_r):.1%}\n")
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Plot Decision Tree and save
plt.figure(figsize=(15, 8))
tree.plot_tree(clf, feature_names=features, filled=True, rounded=True, fontsize=8)
plt.title('ID3 Decision Tree')
plt.savefig('decision_tree_plot.png')
plt.show()

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_plot.png')
plt.show()
