#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn import svm
from sklearn.metrics import confusion_matrix
from data_v1 import make_data

# import random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import EllipticEnvelope

def param(clf=svm.SVC()):
	param_grid = {'C': [0.1, 1, 10], 
					'gamma': [1, 0.1, 0.01],
					'kernel': ['rbf', 'sigmoid','linear']} 

	x_train, y_train, x_test, y_test = make_data()
	# train the model using the training sets
	grid = GridSearchCV(clf, param_grid, refit = True, verbose = 3)

	# fitting the model for grid search

	grid.fit(x_train, y_train)

	# print best parameter after tuning
	print(grid.best_estimator_, grid.best_params_, grid.best_score_, sep=' ')
	

def main(clf,plot=False,savefig=None,replace=False):
	# get data from make_data function as x_train, y_train, x_test, y_test
	x_train, y_train, x_test, y_test = make_data()

	# print x_train
	# print(y_train[0:5])

	# print x_test
	# print(y_test[0:5])

	# create svm classifier
	# clf = svm.SVC(kernel='linear', C=1.0)
	

	# fitting the model for grid search

	clf.fit(x_train, y_train)


	# predict the response for test dataset
	y_pred = clf.predict(x_test)
	if replace:
		y_pred = np.where(y_pred == -1, 0, y_pred)

	# print y_pred
	# print(y_pred)

	# confusion matrix
	cm = confusion_matrix(y_test, y_pred)
	# print(np.unique(y_pred))
	# print(np.unique(y_test))
	# print cm
	# print(cm)

	# accuracy of the model
	print("Accuracy of the model is: ", clf.score(x_test, y_test)*100)

	# plot the confusion matrix
	if plot:
		plt.matshow(cm)
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		if savefig:
			plt.savefig(f'{savefig}.png')
		plt.show()


from sklearn.model_selection import GridSearchCV
if __name__ == '__main__':
	# execute only if run as a script
	# main()
	

	main(clf=svm.SVC(kernel='rbf', C=1.0,gamma=0.1),plot=True)
	
	# param() #SVC(C=1, gamma=0.1) {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'} 0.5087134711332858

	# clf is random forest classifier
	# main(clf=RandomForestClassifier(n_estimators=100),plot=True,savefig='randomforest')
	# main(clf=EllipticEnvelope(contamination=0.01),plot=True,savefig='elliptic',replace=True)

