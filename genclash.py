# import libraries
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from data_v1 import make_data
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from data_v3 import load_spotify_data, split_data_loader, combine_data_loaders, get_data_from_loader
# from data_v2 import make_data

class Net(nn.Module):
	def __init__(self,v1,v2):
		super(Net, self).__init__()
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
	

def load_data(df1, df2):
	# Load the data from the two files
	# df1 = pd.read_csv(file1, usecols=range(10))
	# df2 = pd.read_csv(file2, usecols=range(10))
	
	# Add a label column to each DataFrame
	df1['label'] = 0
	df2['label'] = 1
	# print(df1)
	
	# Concatenate the two DataFrames into one
	# df = pd.concat([df1, df2], ignore_index=True)
	# add the two dataframes together
	# df = df1.append(df2)
	# print(df)
	# append df2 to df1
	df = df1.append(df2)
	# print(df[0:10])
	# replace NaN values with 0
	df = df.fillna(0)

	
	# Convert the DataFrame into a PyTorch tensor
	X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
	y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
	
	# Create a PyTorch DataLoader object
	dataset = TensorDataset(X, y)
	loader = DataLoader(dataset, batch_size=32, shuffle=True)
	
	return loader


def dataloader():
	x_train, y_train, x_test, y_test = make_data()
	x_train = torch.from_numpy(x_train).float()
	y_train = torch.from_numpy(y_train).float()
	x_test = torch.from_numpy(x_test).float()
	y_test = torch.from_numpy(y_test).float()

	trainset = torch.utils.data.TensorDataset(x_train, y_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

	testset = torch.utils.data.TensorDataset(x_test, y_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

	return trainloader, testloader

def preprocess_data(batch_size=32):
	# Load the data from the two files
	gen10 = pd.read_csv('gen1_bottom.csv', usecols=range(10), header=None)
	gen11 = pd.read_csv('gen1_top.csv', usecols=range(10), header=None)
	
	gen1_loader = load_data(gen10, gen11)

	# read gen2 data
	gen20 = pd.read_csv('gen2_bottom.csv', usecols=range(10), header=None)
	gen21 = pd.read_csv('gen2_top.csv', usecols=range(10), header=None)
	
	
	gen2_loader = load_data(gen20, gen21)

	# use load_data with
	gen1_train, gen1_label = get_data_from_loader(gen1_loader)
	gen2_train, gen2_label = get_data_from_loader(gen2_loader)

	# Scale the input data using the standard scaler
	scaler = StandardScaler()
	gen1_train_normalized = torch.tensor(scaler.fit_transform(gen1_train), dtype=torch.float)
	gen1_label_normalized = gen1_label
	gen2_train_normalized = torch.tensor(scaler.fit_transform(gen2_train), dtype=torch.float)
	gen2_label_normalized = gen2_label


	# combine gen1 with label
	gen1_loader=[(gen1_train_normalized[i], gen1_label_normalized[i]) for i in range(len(gen1_train_normalized))]
	# combine gen2 with label
	gen2_loader=[(gen2_train_normalized[i], gen2_label_normalized[i]) for i in range(len(gen2_train_normalized))]

	gen1_loader=DataLoader(gen1_loader, batch_size=batch_size, shuffle=True)
	gen2_loader=DataLoader(gen2_loader, batch_size=batch_size, shuffle=True)

	return gen1_loader, gen2_loader

def engvsindia(batch_size=32):
	# Load the data from the two files
	gen10 = pd.read_csv('data.csv', usecols=range(10), header=None)
	gen11 = pd.read_csv('data_top.csv', usecols=range(10), header=None)
	
	gen1_loader = load_data(gen10, gen11)

	# read gen2 data
	gen20 = pd.read_csv('data_bottom_english.csv', usecols=range(10), header=None)
	gen21 = pd.read_csv('data_top_english.csv', usecols=range(10), header=None)
	
	
	gen2_loader = load_data(gen20, gen21)

	# use load_data with
	gen1_train, gen1_label = get_data_from_loader(gen1_loader)
	gen2_train, gen2_label = get_data_from_loader(gen2_loader)

	# Scale the input data using the standard scaler
	scaler = StandardScaler()
	gen1_train_normalized = torch.tensor(scaler.fit_transform(gen1_train), dtype=torch.float)
	gen1_label_normalized = gen1_label
	gen2_train_normalized = torch.tensor(scaler.fit_transform(gen2_train), dtype=torch.float)
	gen2_label_normalized = gen2_label


	# combine gen1 with label
	gen1_loader=[(gen1_train_normalized[i], gen1_label_normalized[i]) for i in range(len(gen1_train_normalized))]
	# combine gen2 with label
	gen2_loader=[(gen2_train_normalized[i], gen2_label_normalized[i]) for i in range(len(gen2_train_normalized))]

	gen1_loader=DataLoader(gen1_loader, batch_size=batch_size, shuffle=True)
	gen2_loader=DataLoader(gen2_loader, batch_size=batch_size, shuffle=True)

	return gen1_loader, gen2_loader

def train(model,trainloader,validloader,testloader,epochs=100,lr=0.001):
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	train_losses, valid_losses = [], []
	for epoch in range(epochs):
		running_train_loss, running_valid_loss = 0.0, 0.0
		for i, data in enumerate(trainloader):
			inputs, labels = data
			# print(labels)
			# print(inputs.shape)
			labels = labels.unsqueeze(1)
			optimizer.zero_grad()
			# print(inputs.shape)

			# Forward pass
			outputs = model(inputs)
			# print(outputs)
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
				outputs = model(inputs)
				loss = criterion(outputs, labels.float())
				running_valid_loss += loss.item()

		train_loss = running_train_loss / len(trainloader)
		valid_loss = running_valid_loss / len(validloader)

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		# print(f"Epoch {epoch+1}, train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}")



	# Test the model
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			inputs, labels = data
			labels = labels.unsqueeze(1)
			outputs = model(inputs)
			predicted = (outputs >= 0.5).float()
			total += labels.size(0)
			correct += (predicted == labels.float()).sum().item()

	# print(f"Accuracy: {100 * correct/total:.2f}%")
	return 100 * correct/total



def main():
	# load gen1 and gen2 loaders
	gen1_loader, gen2_loader = preprocess_data()
	# for i, data in enumerate(gen2_loader, 0):
	# 	inputs, labels = data
	# 	print(inputs.shape)
	# 	print(labels.shape)
	# 	break

	# train on gen1 and test on gen2
	trainloader,validloader = split_data_loader(gen1_loader)
	# print first sample of trainloader
	# for i, data in enumerate(trainloader, 0):
	# 	inputs, labels = data
	# 	# print(inputs.shape)
	# 	# print(labels.shape)
	# 	break

	testloader = gen2_loader
	v1=32
	v2=16
	train_on_gen1=Net(v1,v2)
	acc1=train(train_on_gen1,trainloader,validloader,testloader,epochs=100,lr=0.001)

	# train on gen2 and test on gen1
	trainloader,validloader = split_data_loader(gen2_loader)
	testloader = gen1_loader
	train_on_gen2=Net(v1,v2)
	acc2=train(train_on_gen2,trainloader,validloader,testloader,epochs=100,lr=0.001)

	print("Accuracy on gen1: ", acc1)
	print("Accuracy on gen2: ", acc2)

def parametertuning():
	# load gen1 and gen2 loaders
	gen1_loader, gen2_loader = preprocess_data()
	# for i, data in enumerate(gen2_loader, 0):
	# 	inputs, labels = data
	# 	print(inputs.shape)
	# 	print(labels.shape)
	# 	break

	# train on gen1 and test on gen2
	trainloader,validloader = split_data_loader(gen1_loader)
	# print first sample of trainloader
	# for i, data in enumerate(trainloader, 0):
	# 	inputs, labels = data
	# 	# print(inputs.shape)
	# 	# print(labels.shape)
	# 	break

	testloader = gen2_loader
	v1 = list(range(32,10,-1))
	v2 = list(range(10,32))
	best_acc = 0
	best_v1 = 0
	best_v2 = 0
	# add the header to the file
	# with open('svm_param_tuning.txt', 'w') as f:
	#     f.write('v1, v2, acc\n')
	for i in v1:
		for j in v2:
			acc = train(Net(v1=i,v2=j),trainloader,validloader,testloader,epochs=100,lr=0.001)
			if acc > best_acc:
				best_acc = acc
				best_v1 = i
				best_v2 = j
			# for each combination of v1 and v2, write the accuracy score to a file along with the values of v1 and v2
			with open('svm_param_tuning_gen1.txt', 'a') as f:
				f.write(f'v1: {i}, v2: {j}, acc: {acc}\n')
	print(f'Best accuracy: {best_acc} with v1: {best_v1} and v2: {best_v2}')


	# load gen1 and gen2 loaders
	gen1_loader, gen2_loader = preprocess_data()
	# for i, data in enumerate(gen2_loader, 0):
	# 	inputs, labels = data
	# 	print(inputs.shape)
	# 	print(labels.shape)
	# 	break

	# train on gen1 and test on gen2
	trainloader,validloader = split_data_loader(gen2_loader)
	# print first sample of trainloader
	# for i, data in enumerate(trainloader, 0):
	# 	inputs, labels = data
	# 	# print(inputs.shape)
	# 	# print(labels.shape)
	# 	break

	testloader = gen1_loader
	v1 = list(range(32,10,-1))
	v2 = list(range(10,32))
	best_acc = 0
	best_v1 = 0
	best_v2 = 0
	# add the header to the file
	# with open('svm_param_tuning.txt', 'w') as f:
	#     f.write('v1, v2, acc\n')
	for i in v1:
		for j in v2:
			acc = train(Net(v1=i,v2=j),trainloader,validloader,testloader,epochs=100,lr=0.001)
			if acc > best_acc:
				best_acc = acc
				best_v1 = i
				best_v2 = j
			# for each combination of v1 and v2, write the accuracy score to a file along with the values of v1 and v2
			with open('svm_param_tuning_gen2.txt', 'a') as f:
				f.write(f'v1: {i}, v2: {j}, acc: {acc}\n')
	print(f'Best accuracy: {best_acc} with v1: {best_v1} and v2: {best_v2}')

def main_engvsindia():
	# load gen1 and gen2 loaders
	gen1_loader, gen2_loader = engvsindia()
	# for i, data in enumerate(gen2_loader, 0):
	# 	inputs, labels = data
	# 	print(inputs.shape)
	# 	print(labels.shape)
	# 	break

	# train on gen1 and test on gen2
	trainloader,validloader = split_data_loader(gen1_loader)
	# print first sample of trainloader
	# for i, data in enumerate(trainloader, 0):
	# 	inputs, labels = data
	# 	# print(inputs.shape)
	# 	# print(labels.shape)
	# 	break

	testloader = gen2_loader
	v1=15
	v2=28
	train_on_gen1=Net(v1,v2)
	acc1=train(train_on_gen1,trainloader,validloader,testloader,epochs=100,lr=0.001)

	# train on gen2 and test on gen1
	trainloader,validloader = split_data_loader(gen2_loader)
	testloader = gen1_loader
	train_on_gen2=Net(v1,v2)
	acc2=train(train_on_gen2,trainloader,validloader,testloader,epochs=100,lr=0.001)

	print("Accuracy on india: ", acc1)
	print("Accuracy on eng: ", acc2)
if __name__ == '__main__':
	# main()
	# parametertuning()
	main_engvsindia()

