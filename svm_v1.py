# import svm from sklearn
from sklearn import svm
# import numpy
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import pandas
import pandas as pd
# import csv
import csv

# read data from csv file
# data = pd.read_csv('data.txt')
# print data
# print(data)

# read data from csv file
# data_top = pd.read_csv('data_top.txt')

# discard from 10th column in both data and data_top
# data = data.iloc[:, 0:10]
# data_top = data_top.iloc[:, 0:10]

# for d in data:
# 	print(d[10])
# 	break
# print data
# print(data[0:5])
# print(data_top[0:5])


#read data.txt and split it with newline first then for each of the data in the list split it with space
data = [line.split() for line in open("data_top_english.txt")]
# print(data[0])
#read data_top.txt and split it with newline first then for each of the data in the list split it with space
data_top = [line.split() for line in open('data_bottom_english.txt')]
# print(data_top[0])

#write data and data_top to a csv file with the name data.csv and data_top.csv
with open('data_top_english.csv', 'w', newline='') as outcsv:
	writer = csv.writer(outcsv)
	# write only the firs 10 columns
	for i in range(len(data)):
		data[i] = data[i][:10]

	writer.writerows(data)

with open('data_bottom_english.csv', 'w', newline='') as outcsv:
	writer = csv.writer(outcsv)
	for i in range(len(data)):
		data[i] = data[i][:10]
	writer.writerows(data_top)
    