# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch


def make_data():
	# read data from data.csv and data_top.csv and store it in data and data_top. remove from 10th column in both data and data_top. No header in both data and data_top
	data_bottom = pd.read_csv('./top_spotify.csv', header=None, usecols=range(21))
	data_top = pd.read_csv('./bottom_spotify.csv', header=None, usecols=range(21))
	# print(data_bottom[0])
	# print the first row of data_bottom
	# print(data_bottom[0:1])
	

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
	x_train = train.drop('class', axis=1)
	y_train = train['class']
	x_test = test.drop('class', axis=1)
	y_test = test['class']

	return x_train.values, y_train.values, x_test.values, y_test.values

if __name__ == '__main__':
	print(make_data())