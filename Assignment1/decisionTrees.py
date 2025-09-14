#-------------------------------------------------------------------------
# AUTHOR: Toni Liang    
# FILENAME: decisionTrees
# SPECIFICATION: 
# This program reads data from a CVS file given by the professor that separates values into 
# categories and gives a target label. Using python, the program can realize a tree structure that is close
# to the decision tree we produced earlier in part A.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 4:52:47
#-----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
#AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays
#importing some Python libraries

# CITATIONS

# learned how to read parse csv files using the following sources:
# Python Tutorial: CSV Module - Comma Separated Values https://www.youtube.com/watch?v=q5uM4VKywbA

# Video on Understanding decision trees using Python with scikit-learn
#  Understanding Decision Trees using Python https://www.youtube.com/watch?v=yi7KsXtaOCo


from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

# encoded features are placed into a mapped dictionary as followed by the video cited above

age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'No': 1, 'Yes': 2}
tear_map = {'Reduced': 1, 'Normal': 2}



#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append(row)


#encode the original categorical training features into numbers and add to the 4D
#array X.

#--> add your Python code here
for row in db:
    encoded_row = [
        age_map[row[0]],             # Age
        spectacle_map[row[1]],       # Spectacle Prescription
        astigmatism_map[row[2]],     # Astigmatism
        tear_map[row[3]]             # Tear Production Rate
    ] # X =
    X.append(encoded_row)


#encode the original categorical training classes into numbers and add to the
#vector Y.


#--> addd your Python code here
label_map = {'Yes': 1, 'No': 2}
for row in db: # Y = 
    Y.append(label_map[row[4]])

#fitting the decision tree to the data using entropy as your impurity measure
#--> addd your Python code here
#clf =

clf = tree.DecisionTreeClassifier(criterion='entropy') # this means that we want to use entrophy to help split data
clf = clf.fit(X, Y) # training clf by using encoded features and class labels 

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'],
class_names=['Yes','No'], filled=True, rounded=True)
plt.show()