# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from data_v1 import make_data
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
# from data_v2 import make_data



# Define the neural network architecture
class Net(nn.Module):
	def __init__(self,v1,v2):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(10, v1)
		self.fc2 = nn.Linear(v1, v2)
		# self.fc3 = nn.Linear(16, 8)
		self.fc4 = nn.Linear(v2, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		# x = torch.relu(self.fc3(x))
		x = self.sigmoid(self.fc4(x))
		return x
	

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(10, 10)
		self.fc2 = nn.Linear(10, 10)
		# self.fc3 = nn.Linear(10, 10)
		# self.fc4 = nn.Linear(10, 10)
		self.fc5 = nn.Linear(10, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		# x = torch.relu(self.fc3(x))
		# x = torch.relu(self.fc4(x))
		x = self.sigmoid(self.fc5(x))
		return x

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=2)
		self.conv2 = nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=2)
		self.conv3 = nn.Conv1d(8, 10, kernel_size=3, stride=1, padding=2)
		self.conv4 = nn.Conv1d(10, 8, kernel_size=3, stride=1, padding=2)
		self.conv5 = nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=2)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.mean = nn.AvgPool1d(kernel_size=2)
		self.fc1 = nn.Linear(8, 10)
		self.fc2 = nn.Linear(10, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# x = x.view(32,10,1)  # add new channel dimension
		x = F.relu(self.conv1(x))
		x = self.pool(x)
		x = F.relu(self.conv2(x))
		x = self.pool(x)
		x = F.relu(self.conv3(x))
		x = self.pool(x)
		x = F.relu(self.conv4(x))
		x = self.pool(x)
		x = F.relu(self.conv5(x))
		x = self.pool(x)
		x = x.view(x.shape[0], 8)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		x = self.sigmoid(x)
		return x






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


epochs=100
lr=0.001
# get the trainloader and testloader
trainloader, validloader, testloader = preprocess_data()


# print(trainloader.dataset[0][0].shape)
# Initialize the network
# net = Net()
	# net = MLP()
def cnnshit():
	net = CNN()

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
			inputs=inputs.reshape( (inputs.shape[0],1,10) )
			# print(inputs.shape)

			# Forward pass
			outputs = net(inputs)
			# print(outputs.shape, labels.shape)
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
				inputs=inputs.reshape( (inputs.shape[0],1,10) )
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
			inputs=inputs.reshape( (inputs.shape[0],1,10) )
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

def main(v1,v2):
	
	# Initialize the network
	net = Net(v1,v2)
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

		# print(f"Epoch {epoch+1}, train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}")



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
			
	acc = 100 * correct/total
	# print(f"Accuracy: {100 * correct/total:.2f}%")

	# Plot the training and validation loss
	# plt.plot(train_losses, label='Training Loss')
	# plt.plot(valid_losses, label='Validation Loss')
	# plt.legend()
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.show()
	return acc


def svm_param_tuning():
	v1 = list(range(10, 32))
	v2 = list(range(10, 32))
	best_acc = 0
	best_v1 = 0
	best_v2 = 0
	# add the header to the file

	for i in v1:
		for j in v2:
			acc = main(i, j)
			if acc > best_acc:
				best_acc = acc
				best_v1 = i
				best_v2 = j
			# for each combination of v1 and v2, write the accuracy score to a file along with the values of v1 and v2
			with open('svm_param_tuning_v1.txt', 'a') as f:
				f.write(f'v1: {i}, v2: {j}, acc: {acc}\n')
	print(f'Best accuracy: {best_acc} with v1: {best_v1} and v2: {best_v2}')

if __name__ == '__main__':
	svm_param_tuning()
	# main()