import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim


'''
Deep AutoEncoder using custom class (inheriting nn.Module) in pytorch. Tied
weights are used, unlike deep_autoencoder_pt.py which learns separate weights
on the encoder and decoder sides.
'''


class DeepAutoEncoderModule(nn.Module):

    def __init__(self, nodes, D, activation='relu'):
        super(DeepAutoEncoderModule, self).__init__()
        self.nodes = nodes
        self.D = D
        if activation == 'relu':
            self.act = nn.functional.relu
        elif activation == 'tanh':
            self.act = torch.tanh
        else:
            self.act = torch.sigmoid
        self.build()

    def build(self):
        M1 = self.D
        'Encoder layers'
        self.dropout = nn.Dropout(p=.5)
        self.weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()
        self.hidden_bnorms = nn.ModuleList()
        for i, M2 in enumerate(self.nodes):
            self.weights.append(nn.Parameter(torch.randn(M1, M2)))
            self.hidden_biases.append(nn.Parameter(torch.zeros(M2)))
            self.hidden_bnorms.append(nn.BatchNorm1d(M2))
            M1 = M2
        'Decoder layers (weights are tied, so only biases)'
        self.out_biases = nn.ParameterList()
        self.out_bnorms = nn.ModuleList()
        for M in reversed([self.D] + self.nodes[:-1]):
            self.out_biases.append(nn.Parameter(torch.zeros(M)))
            self.out_bnorms.append(nn.BatchNorm1d(M))

    def encode(self, X):
        X = self.dropout(X)
        for W, b, bnorm in zip(
                self.weights, self.hidden_biases, self.hidden_bnorms):
            X = X @ W + b
            X = bnorm(X)
            X = self.act(X)
        return X

    def decode(self, X):
        for W, b, bnorm in zip(
                reversed(self.weights), self.out_biases, self.out_bnorms):
            X = X @ W.transpose(0, 1) + b
            X = bnorm(X)
            X = self.act(X)
        return X

    def forward(self, X):
        return self.decode(self.encode(X))


class DeepAutoEncoder(object):

    def __init__(self, nodes, activation):
        self.nodes = nodes
        self.act = activation

    def fit(self, X, lr=1e-4, epochs=40, batch_sz=200, print_every=50):

        N, D = X.shape
        X = torch.from_numpy(X).float().to(device)

        self.model = DeepAutoEncoderModule(self.nodes, D, activation=self.act)
        self.model.to(device)

        self.loss = nn.MSELoss().to(device)
        # self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            print("epoch:", i, "n_batches:", n_batches)
            inds = torch.randperm(X.size()[0])
            X = X[inds]
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

                cost = self.train(Xbatch)  # train

                if j % print_every == 0:
                    print("cost: %f" % (cost))
                    costs.append(cost)
            # costs.append(cost)

        plt.plot(costs)
        plt.show()

    def train(self, inputs):
        # set the model to training mode
        # dropout and batch norm behave differently in train vs eval modes
        self.model.train()

        # commented out but not deleted as a reminder, Variable wrapping of
        # tensors for forward operations is deprecated. Tensors work on their
        # own now, and have requires_grad=False by default.
        # inputs = Variable(inputs, requires_grad=False)

        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        reconstruction = self.model.forward(inputs)
        output = self.loss.forward(reconstruction, inputs)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    # similar to train() but not doing the backprop step
    def get_cost(self, inputs):
        # set the model to testing mode
        # dropout and batch norm behave differently in train vs eval modes
        self.model.eval()

        'Wrap in no_grad to prevent graph from storing info for backprop'
        with torch.no_grad():
            # Variable wrapping is deprecated (no longer required)
            # inputs = Variable(inputs, requires_grad=False)

            # Forward
            reconstruction = self.model.forward(inputs)
            output = self.loss.forward(reconstruction, inputs)

        return reconstruction, output.item()

    def get_reduced(self, X):
        'Return reduced dimensionality hidden representation of X'
        self.model.eval()
        X = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            # Variable wrapping is deprecated (no longer required)
            # inputs = Variable(X, requires_grad=False)
            reduced = self.model.encode(X).cpu().numpy()
        return reduced

    def reconstruct(self, X):
        for i in range(50):
            idx = np.random.randint(0, X.shape[0], 1)
            sample = torch.from_numpy(X[idx]).float().to(device)

            self.model.eval()

            with torch.no_grad():
                # Variable wrapping is deprecated (no longer required)
                # inputs = Variable(sample, requires_grad=False)

                # Forward
                reconstruction = self.model.forward(sample).cpu().numpy()

            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(X[idx].reshape(28, 28), cmap='gray')
            axes[1].imshow(reconstruction.reshape(28, 28), cmap='gray')
            plt.show()

            again = input(
                "Show another reconstruction? Enter 'n' to quit\n")
            if again == 'n':
                break


# use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
