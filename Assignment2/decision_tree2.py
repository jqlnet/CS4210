#-------------------------------------------------------------------------
# AUTHOR: Toni Liang
# FILENAME:Decision_Tree part 2
# SPECIFICATION: This is the second assignment from machine learning 4210 where the objective of the program 
# is to create a training model of the decision tree by using 3 cvs training files that contain data on contact lenses.
# the program shall then train 10 iterations of each dataset and return the average accuracy of the data given.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd


# given predefined datasets 
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())


# used mapping here to transform training features into numbers
age_map = {'Young':1, 'Prepresbyopic':2, 'Presbyopic':3}
spect_map = {'Myope':1, 'Hypermetrope':2}
astig_map = {'Yes':1, 'No':2}
tear_map = {'Normal':1, 'Reduced':2}
class_map = {'Yes':1, 'No':2}

for ds in dataSets:

    # dbTraining = [] reading training data directly into pandas using the dbTraining = pd.read_cvs(ds), no longer need
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here, ds is the current training file name/title
    dbTraining = pd.read_csv(ds)

    for _, row in dbTraining.iterrows():
        features = [
            age_map[row['Age']],
            spect_map[row['Spectacle Prescription']],
            astig_map[row['Astigmatism']],
            tear_map[row['Tear Production Rate']]
        ]

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
        X.append(features)

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
        Y.append(class_map[row['Recommended Lenses']])

    # this is the LIST of accuracies after the 10 iterations we will use later
    accuracies = []
    #Loop your training and test tasks 10 times here
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> add your Python code here, classifer is called here:
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
       # no need to add it here, because i stated it at the top of the program.

       #the correct resets to 0 EACH iteration, but it is essentially the starting accuracy for each loop.
       correct = 0

       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            test_features = [
                age_map[data[0]],
                spect_map[data[1]],
                astig_map[data[2]],
                tear_map[data[3]]
            ]
            class_predicted = clf.predict([test_features])[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            true_label = class_map[data[4]]
            if class_predicted == true_label:
                correct += 1

    accuracy = correct / len(dbTest) # the total
    accuracies.append(accuracy) # appends to the accuracy list on top of the program

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here, to get the avg accuracy, we divide the sum/total of the accuracies by however many runs or the length of the accuracies list
    avg_accuracy = sum(accuracies) / len(accuracies)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here, the ds will print the file we are currently training with. e.g., contact_lens_training_1, 2, 3, etc...
    print(f"Final accuracy when training on {ds}: {round(avg_accuracy, 2)}")
