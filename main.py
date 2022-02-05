from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split

# Using pandas to read CSV file
data = pd.read_csv("Q1.csv")

# Setting up dependent and independent variables eg. x and y axis
# Timestamp was dropped as it's irrelevant to the problem
x = data.drop(['Fault', 'Timestamp'], axis='columns')
y = data['Fault']

# Splitting the shuffled data set into 50% test and train data
# Cross fold validation was considered but avoided due to large data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=True)

# The decision tree classifier model is chosen and is fitted with the training values from earlier
model = tree.DecisionTreeClassifier()
model.fit(x_train.values, y_train.values)

# Model is scored based on test data and printed
print(f"Model Score: {model.score(x_test.values, y_test.values)}\n")

# Application - 1c
# Taking inputs
ch1t1 = input("Chiller Pump 1 Temperature 1: ")
ch1t2 = input("Chiller Pump 1 Temperature 2: ")

ch2t1 = input("Chiller Pump 2 Temperature 1: ")
ch2t2 = input("Chiller Pump 2 Temperature 2: ")

ch1v1 = input("Chiller Pump 1 Vibration Sensor Data 1: ")
ch1v2 = input("Chiller Pump 1 Vibration Sensor Data 2: ")

ch2v1 = input("Chiller Pump 2 Vibration Sensor Data 1: ")
ch2v2 = input("Chiller Pump 2 Vibration Sensor Data 2: ")

# Making prediction based on model
predFault = model.predict([[ch1t1, ch1t2, ch2t1, ch2t2, ch1v1, ch1v2, ch2v1, ch2v2]])

print(f"\nPredicted Fault (Output): {predFault[0]}")
