import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Define neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # fully connected layers
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)  # PyTorch: drop probability

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


# Prepare dataset
def prepareDataA4(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    y = data[:, 0]
    X = data[:, 1:] / 255.  # (10000, 2352) --> 2352 = 84 * 28 --> one image is 28*28 --> 28*3 = 84
    return X, y


# takes an n × d input matrix X and an n × 1 label vector y (where each entry is
# an integer between 0 and 9, inclusive, representing the image labels), and returns
# your model of any type
def learn(X, y):
    tensor_X = torch.Tensor(X)
    tensor_y = torch.LongTensor(y)

    dataset = TensorDataset(tensor_X, tensor_y)  # create PyTorch TensorDataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # create PyTorch DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)  # use proper device

    # Optimizer: AdaGrad
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)

    # Training
    n_epochs = 30
    for epoch in range(n_epochs):
        train(model, optimizer, dataloader, device)

    return model


# takes an m × d input matrix Xtest and a model learned by your algorithm, and returns an m × 1 prediction vector yhat
def classify(Xtest, model):
    img1 = Xtest[:, 0:784]
    img2 = Xtest[:, 784:1568]
    img3 = Xtest[:, 1568:2352]
    img1 = torch.Tensor(img1)
    img2 = torch.Tensor(img2)
    img3 = torch.Tensor(img3)

    model.eval()
    with torch.no_grad():
        output = model(img1)
        predNum = output.argmax(dim=1).numpy()
        actualImg = np.zeros(img1.shape)
        for b in range(output.shape[0]):
            if predNum[b] % 2 == 0:  # EVEN
                actualImg[b] = img3[b]
            else:  # ODD
                actualImg[b] = img2[b]

        actualImg = torch.Tensor(actualImg)
        output = model(actualImg)

    yhat = output.argmax(dim=1, keepdim=True)
    yhat = np.array(yhat).flatten()
    return yhat


# Training loop
def train(model, optimizer, train_dataloader, device):
    model.train()  # entering training mode (dropout behaves differently)

    # takes first 28 rows, so the first number
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)  # move data to the same device

        img1 = data[:, 0:784]
        img2 = data[:, 784:1568]
        img3 = data[:, 1568:2352]

        optimizer.zero_grad()  # clear existing gradients
        output = model(img1)  # forward pass
        predNum = output.argmax(dim=1).numpy()

        actualImg = np.zeros(img1.shape)
        for b in range(output.shape[0]):
            if predNum[b] % 2 == 0:  # EVEN
                actualImg[b] = img3[b]
            else:  # ODD
                actualImg[b] = img2[b]

        actualImg = torch.Tensor(actualImg)
        optimizer.zero_grad()  # clear existing gradients
        output = model(actualImg)  # forward pass

        loss = F.nll_loss(output, target)  # compute the loss
        loss.backward()  # backward pass: calculate the gradients
        optimizer.step()  # take a gradient step


if __name__ == '__main__':
    # testLRMomentumOptimizer()
    # testBatchSize()
    # testLRAdagrad()

    # IMPORTANT
    X, y = prepareDataA4('A4data/A4train.csv')
    Xtest, ytest = prepareDataA4('A4data/A4val.csv')
    model = learn(X, y)

    start_time = time.time()
    yhat = classify(Xtest, model)
    print(f"Classifying Time: {time.time() - start_time:.3f}")

    # compare yhat and ytest to calculate accuracy
    ytest = np.array(ytest, dtype=int)
    print(f"    Actual: {ytest}")
    print(f"Prediction: {yhat}")

    print(f"Accuracy: {((ytest == yhat).sum() / len(ytest))}")


# plot images functions
def plot3Img(X, y, index):  # plots 3 images
    img = X[index].reshape((84, 28))
    plt.imshow(img, cmap='gray')
    plt.title(str(y[index]))
    plt.show()
    return


def plot1Img(X, y, index, title):
    img = X[index].reshape(28, 28)
    tmp = str(y[index]) + " " + title
    plt.title(tmp)
    plt.imshow(img, cmap="gray")
    plt.show()


def plot1ImagePred(X, output):
    img = X.reshape(28, 28)
    plt.title(output)
    plt.imshow(img, cmap="gray")
    plt.show()


def learnDiffParams(X, y, model, o, device, n_epochs, b):
    tensor_X = torch.Tensor(X)
    tensor_y = torch.LongTensor(y)

    dataset = TensorDataset(tensor_X, tensor_y)  # create PyTorch TensorDataset
    dataloader = DataLoader(dataset, batch_size=b, shuffle=True)  # create PyTorch DataLoader

    optimizer = o

    # Training
    for epoch in range(n_epochs):
        train(model, optimizer, dataloader, device)

    return model


def testLRMomentumOptimizer():
    X, y = prepareDataA4('A4data/A4train.csv')
    Xtest, ytest = prepareDataA4('A4data/A4val.csv')

    # test diff learning rates and momentum for optimizer
    learningRates = [0.5, 0.25, 0.1, 0.05, 0.001]
    momentum = [0, 0.25, 0.5, 0.6, 0.75, 0.9]
    n_epochs = 30
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for l in range(len(learningRates)):
        for m in range(len(momentum)):
            model = Net().to(device)  # use proper device
            optimizer = optim.SGD(model.parameters(), lr=learningRates[l], momentum=momentum[m])
            model = learnDiffParams(X, y, model, optimizer, device, n_epochs, batch_size)
            yhat = classify(Xtest, model)

            # compare yhat and ytest to calculate accuracy
            ytest = np.array(ytest, dtype=int)
            accuracy = ((ytest == yhat).sum() / len(ytest))
            print(f"Accuracy for learning rate {learningRates[l]} and momentum {momentum[m]}: {accuracy}")


def testLRAdagrad():
    X, y = prepareDataA4('A4data/A4train.csv')
    Xtest, ytest = prepareDataA4('A4data/A4val.csv')

    # test diff learning rates and momentum for optimizer
    learningRates = [0.5, 0.25, 0.1, 0.05, 0.001]
    n_epochs = 30
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for l in range(len(learningRates)):
        model = Net().to(device)  # use proper device
        optimizer = optim.Adagrad(model.parameters(), lr=learningRates[l])
        model = learnDiffParams(X, y, model, optimizer, device, n_epochs, batch_size)
        yhat = classify(Xtest, model)
        # compare yhat and ytest to calculate accuracy
        ytest = np.array(ytest, dtype=int)
        accuracy = ((ytest == yhat).sum() / len(ytest))
        print(f"Accuracy for learning rate {learningRates[l]}: {accuracy}")


def testBatchSize():
    X, y = prepareDataA4('A4data/A4train.csv')
    Xtest, ytest = prepareDataA4('A4data/A4val.csv')

    # test diff learning rates and momentum for optimizer
    batchsizes = [2, 4, 8, 16, 32, 64, 128]
    epochs = [5, 15, 30, 60]
    learningRate = 0.01
    # momentum
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for e in range(len(epochs)):
        for b in range(len(batchsizes)):
            model = Net().to(device)  # use proper device
            optimizer = optim.Adagrad(model.parameters(), lr=learningRate)
            model = learnDiffParams(X, y, model, optimizer, device, epochs[e], batchsizes[b])
            yhat = classify(Xtest, model)

            # compare yhat and ytest to calculate accuracy
            ytest = np.array(ytest, dtype=int)
            accuracy = ((ytest == yhat).sum() / len(ytest))
            print(f"Accuracy for {epochs[e]} epochs and with batch size {batchsizes[b]}: {accuracy}")
