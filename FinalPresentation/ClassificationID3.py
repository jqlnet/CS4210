# Comprehensive Decision Tree Evaluation Script for PriceRunner Dataset
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# the file to be opened
csv_file = 'pricerunner_aggregate.csv'

# Read data
df = pd.read_csv(csv_file)

# Encode categorical columns as integers
df['Product Title'] = df['Product Title'].astype('category').cat.codes
df['Merchant ID'] = df['Merchant ID'].astype(int)
df['Category Label'] = df['Category Label'].astype('category').cat.codes

# You can add more features as needed (e.g., 'Cluster Label', 'Category ID')
features = ['Product Title', 'Merchant ID']  # add more features to this list if relevant
target = 'Category Label'

X = df[features]
y = df[target]

# Split into training and testing sets (80/20 split, randomize for fairness)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train ID3 Decision Tree (using entropy)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Metrics and Reporting
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')
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
