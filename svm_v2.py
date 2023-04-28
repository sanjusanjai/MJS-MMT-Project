#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import svm
from sklearn.metrics import confusion_matrix


# read data from data.csv and data_top.csv and store it in data and data_top. remove from 10th column in both data and data_top. No header in both data and data_top
data_bottom = pd.read_csv('data.csv', header=None, usecols=range(10))
data_top = pd.read_csv('data_top.csv', header=None, usecols=range(10))

# replace Nan with 0
data_bottom = data_bottom.fillna(0)
data_top = data_top.fillna(0)

# convert values to fit into float64
data_bottom = data_bottom.astype('float64')
data_top = data_top.astype('float64')

# print data
# print(data_bottom[0:5])
# print(data_top[0:5])

#make data_top as 1 and data as 0
data_top['class'] = 1
data_bottom['class'] = 0

# combine data_bottom and data_top to data
data = pd.concat([data_bottom, data_top], ignore_index=True)

# print data
# print(data[0:5])

# shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# print data
# print(data[0:5])

# split data into train and test with percentage
train = data.sample(frac=0.9)
test = data.drop(train.index)

# print train
# print(train[0:5])
# print(len(train))

# print test
# print(test[0:5])
# print(len(test))

# split train and test into x and y
x_train = train.iloc[:, 0:10]
y_train = train.iloc[:, 10]
x_test = test.iloc[:, 0:10]
y_test = test.iloc[:, 10]

# print x_train
# print(x_train[0:5])
# print(len(x_train))

# print y_train
# print(y_train[0:5])
# print(len(y_train))

# fit the model
clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

# print clf
# print(clf)

# predict the model
y_pred = clf.predict(x_test)

# print y_pred
# print(y_pred)

# accuracy of the model
print("Accuracy of the model is: ", clf.score(x_test, y_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
