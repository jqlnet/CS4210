#-------------------------------------------------------------------------
# AUTHOR: Toni Liang
# FILENAME: knn.py
# SPECIFICATION: This program will read the file email_classification.csv and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task
# The dataset consists of email samples, where each sample includes the counts of specific words, representing their frequency of occurence.
# FOR: CS 4210- Assignment #2 K Nearest Neighbor
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

errors = 0 # resets errors to 0
n = len(db) # length of database

# useful labeling found on youtube video on KNN encoding
def encode_label(label):
    if label == "ham":
        return 0
    else:
        return 1

#Loop your data to allow each instance to be your test set
for i in range(n): 
# for i in db iterates directly over elements in the rows
# for i in range(n) allows us to put an element i in each iteration and lets us remove them or use them as well.

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = [row[:-1] for idx, row in enumerate(db) if idx != i]
    X = [list(map(float, x)) for x in X]

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = [encode_label(row[-1]) for idx, row in enumerate(db) if idx != i]

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(x) for x in db[i][:-1]]
    testLabel = encode_label(db[i][-1])

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

#Compare the prediction with the true label of the test instance to start calculating the error rate.
#--> add your Python code here
error_rate = errors / n

#Print the error rate
#--> add your Python code here
print(f"LOO-CV error rate for 1NN: {error_rate:.4f}")

