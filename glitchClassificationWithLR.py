from __future__ import print_function
import numpy as np
import csv

finalVals = []

def sigmoid(t):
    return 1/(1 + np.exp(-t))

def checkSize(w, X, y):
	# w and y are column vector, shape [N, 1] not [N,]
	# X is a matrix where rows are data sample
	assert X.shape[0] == y.shape[0]
	assert X.shape[1] == w.shape[0]
	assert len(y.shape) == 2
	assert len(w.shape) == 2
	assert w.shape[1] == 4
	assert y.shape[1] == 4

def compactNotation(X):
	return np.hstack([np.ones([X.shape[0], 1]), X])

def readData(path):
	"""
	Read data from path (either path to MNIST train or test)
	return X in compact notation (has one appended)
	return Y in with shape [10000,1] and starts from 0 instead of 1
	"""
	reader = csv.reader(open(path, "rb"), delimiter=",")
	d = list(reader)
	# import data and reshape appropriately
	data = np.array(d).astype("float")
	X = data[:,:-2]
	y = data[:,-1]
	y = y-1
	y.shape = (len(y),1)  
	# pad data with ones for more compact gradient computation
	X = compactNotation(X)
	return X,y


def softmaxGrad(w, X, y):
    checkSize(w, X, y)
    X = X.transpose()
	### RETURN GRADIENT
    prod1 = np.dot(X.transpose(), w)
    #print prod1.shape, "is prod1's shape"
    #print y.shape, "is y's shape"
    negY = -1*y
    prod2 = np.multiply(negY, prod1)
    #print prod2.shape, "is prod2" 
    sigm = sigmoid(prod2)
    # print sigm
    prod3 = np.multiply(negY, sigm)
    # print prod3 
    #print prod3.shape
    prod4 = np.dot(X, prod3)
    #print prod4.shape
    return prod4

def accuracy(OVA, X, y):
	"""
	Calculate accuracy using matrix operations!
	"""
	correct = 0.0
	n = float(y.shape[0])
	Yout = np.empty_like(y)
	for i in range(len(y)):
		x = X[i,:]
		out = np.dot(OVA.transpose(),x)
		print (np.argmax(out))
		finalVals.append(np.argmax(out))
		# print (y[i][0])
		print ("--------")
		if int(y[i][0]) == np.argmax(out):
			correct += 1
	# print(correct/n)

def gradientDescent(grad, w0, *args, **kwargs):
	max_iter = 1000
	alpha = 0.001
	eps = 10^(-5)

	w = w0
	iter = 0
	while True:
		gradient = grad(w, *args)
		w = w - alpha * gradient
		if iter > max_iter or np.linalg.norm(gradient) < eps:
			break
		if iter  % 1000 == 1:
			print("Iter %d " % iter)
		iter += 1
	return w


def oneVersusAll(Y, value):
	"""
	generate label Yout, 
	where Y == value then Yout would be 1
	otherwise Yout would be -1
	"""
	newY = np.empty_like(Y)
	for i in range(len(Y)):
		if Y[i] == value:
			newY[i] = 1
		else:
			newY[i] = -1
	return newY	

def writeVals(path):
	reader = csv.reader(open(path, "rb"), delimiter=",")
	d = list(reader)
	# import data and reshape appropriately
	data = np.array(d).astype("float")
	data[:, -1] = finalVals
	with open("opfile.csv", 'a') as writeFile:
		spamwriter = csv.writer(writeFile, delimiter=',')
		for row in data:
			spamwriter.writerow(row)


if __name__=="__main__":

	trainX, trainY = readData('train_data_label.csv')
	# # training individual classifier
	Nfeature = trainX.shape[1]
	Nclass = 4
	OVA = np.zeros((Nfeature, Nclass))
	Yclass = np.zeros((trainY.shape[0], Nclass))

	for i in range(Nclass):
		Yclass[:,i] = oneVersusAll(trainY, i)[:,0]

	w0 = np.zeros((Nfeature, Nclass))
	OVA = gradientDescent(softmaxGrad, w0, trainX, Yclass)
	# print('here')
	print ("Accuracy for training set is")
	# accuracy(OVA, trainX, trainY)
	testX, testY = readData('test_data.csv')
	print ("Accuracy for the test set is")
	accuracy(OVA,testX,testY)
	print (finalVals)

	writeVals('test_data.csv')
