#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch.optim as optim

from sklearn import svm
from sklearn.metrics import confusion_matrix
from data_v1 import make_data

# import random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset

from data_v3 import load_spotify_data, split_data_loader, combine_data_loaders, get_data_from_loader
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler


def preprocess_data(batch_size=32, val_split=0.05):
    x_train, y_train, x_test, y_test = make_data()

    # Scale the input data using the standard scaler
    scaler = StandardScaler()
    x_train_normalized = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float)
    x_test_normalized = torch.tensor(scaler.transform(x_test), dtype=torch.float)

    # Split the training data into training and validation sets
    from sklearn.model_selection import train_test_split
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train_normalized, y_train, test_size=val_split, random_state=42)

    # Combine the input and target data into tuples
    train_data = [(x_train_split[i], y_train_split[i]) for i in range(len(x_train_split))]
    val_data = [(x_val_split[i], y_val_split[i]) for i in range(len(x_val_split))]
    test_data = [(x_test_normalized[i], y_test[i]) for i in range(len(x_test))]

    # Create DataLoader objects for the train, validation, and test data
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return trainloader, valloader, testloader


class SVM(nn.Module):
	def __init__(self,v1,v2):
		super(SVM, self).__init__()
		self.fc1 = nn.Linear(10, v1)
		self.fc2 = nn.Linear(v1, v2)
		# self.fc3 = nn.Linear(v2, 8)
		self.fc4 = nn.Linear(v2, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		# x = torch.relu(self.fc3(x))
		x = self.sigmoid(self.fc4(x))
		return x

def main_old(clf=svm.SVC(kernel='linear', C=1.0),plot=False,savefig=None):
	# get data from make_data function as x_train, y_train, x_test, y_test
	# x_train, y_train, x_test, y_test = make_data()

	top_english="data_top_english.csv"
	bottom_english="data_bottom_english.csv"

	# load data
	english_top_loader = load_spotify_data(top_english,label=1)
	english_bottom_loader = load_spotify_data(bottom_english,label=0)

	# combine data to one loader and then split the data
	english_loader = combine_data_loaders(english_top_loader,english_bottom_loader)
	train_loader, test_loader = split_data_loader(english_loader)

	# from the loaders get the x_train, y_train, x_test, y_test
	# train_indices = train_loader.sampler.indices
	# test_indices = test_loader.sampler.indices
	# x_train, y_train = train_loader.dataset.data[train_indices], train_loader.dataset.targets[train_indices]
	# x_test, y_test = test_loader.dataset.data[test_indices], test_loader.dataset.targets[test_indices]
	# get data using the function get_data_from_loader
	x_train, y_train = get_data_from_loader(train_loader)
	x_test, y_test = get_data_from_loader(test_loader)



	# print x_train
	# print(y_train[0:5])

	# print x_test
	# print(y_test[0:5])

	# create svm classifier
	# clf = svm.SVC(kernel='linear', C=1.0)

	# train the model using the training sets
	clf.fit(x_train, y_train)

	# predict the response for test dataset
	y_pred = clf.predict(x_test)

	# print y_pred
	# print(y_pred)

	# confusion matrix
	cm = confusion_matrix(y_test, y_pred)

	# print cm
	print(cm)

	# accuracy of the model
	print("Accuracy of the model is: ", clf.score(x_test, y_test))

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

def main_english(net, epochs=100, lr=0.001):
	top_english="data_top_english.csv"
	bottom_english="data_bottom_english.csv"

	# load data
	english_top_loader = load_spotify_data(top_english,label=1)
	english_bottom_loader = load_spotify_data(bottom_english,label=0)

	# combine data to one loader and then split the data
	english_loader = combine_data_loaders(english_top_loader,english_bottom_loader)
	# train_loader, test_loader = split_data_loader(english_loader)
	trainloader, validloader, testloader = preprocess_data()
	# from the loaders get the x_train, y_train, x_test, y_test
	# train_indices = train_loader.sampler.indices
	# test_indices = test_loader.sampler.indices
	# x_train, y_train = train_loader.dataset.data[train_indices], train_loader.dataset.targets[train_indices]
	# x_test, y_test = test_loader.dataset.data[test_indices], test_loader.dataset.targets[test_indices]
	# # get data using the function get_data_from_loader
	# x_train, y_train = get_data_from_loader(train_loader)
	# x_test, y_test = get_data_from_loader(test_loader)
	# Initialize the network
	# net = SVM()
	# net = MLP()

	# Define the loss function and optimizer
	criterion = nn.BCELoss()
	optimizer = optim.Adam(net.parameters(), lr=lr)

	train_losses, valid_losses = [], []
	for epoch in range(epochs):
		running_train_loss, running_valid_loss = 0.0, 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			labels = labels.unsqueeze(1)
			optimizer.zero_grad()
			# print(inputs.shape)

			# Forward pass
			outputs = net(inputs)
			loss = criterion(outputs, labels.float())

			# Backward and optimize
			loss.backward()
			optimizer.step()

			running_train_loss += loss.item()

		# Compute validation loss
		with torch.no_grad():
			for i, data in enumerate(validloader, 0):
				inputs, labels = data
				labels = labels.unsqueeze(1)
				outputs = net(inputs)
				loss = criterion(outputs, labels.float())
				running_valid_loss += loss.item()

		train_loss = running_train_loss / len(trainloader)
		valid_loss = running_valid_loss / len(validloader)

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		print(f"Epoch {epoch+1}, train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}")



	# Test the model
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			inputs, labels = data
			labels = labels.unsqueeze(1)
			outputs = net(inputs)
			predicted = (outputs >= 0.5).float()
			total += labels.size(0)
			correct += (predicted == labels.float()).sum().item()

	print(f"Accuracy: {100 * correct/total:.2f}%")

	# Plot the training and validation loss
	plt.plot(train_losses, label='Training Loss')
	plt.plot(valid_losses, label='Validation Loss')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()



if __name__ == '__main__':
	# execute only if run as a script
	# main()
	# main(clf=svm.SVC(kernel='rbf', C=1.0),plot=True)

	# clf is random forest classifier
	# main(clf=RandomForestClassifier(n_estimators=100),plot=True,savefig='randomforest')
	v1=16
	v2=12
	net=SVM(v1=v1,v2=v2)
	main_english(net=net,epochs=200,lr=0.001)
