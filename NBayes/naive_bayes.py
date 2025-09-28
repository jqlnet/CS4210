#-------------------------------------------------------------------------
# AUTHOR: Toni Liang
# FILENAME: native_bayes.py
# SPECIFICATION: For this program, we will be reading the file weather_training.csv training set and outputting the classification of each of the 10 instances
# from the file weather_test test set and if the classification confidence is greater than or equal to 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#here we have the mappings that transform the features into numbers usually down there but the python code is fine up here too.
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temp_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Weak': 1, 'Strong': 2}

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for row in dbTraining:
    X.append([
        outlook_map[row[1]],
        temp_map[row[2]],
        humidity_map[row[3]],
        wind_map[row[4]]
    ])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
y_map = {'Yes': 1, 'No': 2}
Y = []
for row in dbTraining:
    Y.append(y_map[row[5]])

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here, we will be using the GaussianNB since it was included in our library.
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print("Day,Outlook,Temperature,Humidity,Wind,PlayTennis,Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

#here we will convert each test sample feature again
for row in dbTest:
    x_test = [
        outlook_map[row[1]],
        temp_map[row[2]],
        humidity_map[row[3]],
        wind_map[row[4]]
    ]

    #now by using the predict proba, we can predict probability of each class using that specific sample
    proba = clf.predict_proba([x_test])[0]
    #predict gives you most likely class
    pred_class_num = clf.predict([x_test])[0]
    class_map_rev = {1: 'Yes', 2: 'No'}
    pred_class = class_map_rev[pred_class_num]
    #the confidence value is here which is if the predicted class is 1, then the proba will be equal to the confidence
    confidence = proba[pred_class_num - 1]
    #if the confidence is >= 0.75 then we will print out else its not confident to give out the answer.
    if confidence >= 0.75:
        print(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{pred_class},{confidence:.2f}")
