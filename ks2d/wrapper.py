import torch

class KS2DModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.X_min = 0
        self.X_max = 1

    def setup_normalization(self, X):
        mean = X.mean(dim=[1,2])
        self.X_min = (X - mean).min()
        self.X_max = (X - mean).max()

    def train(self, X, Y, **kwargs):

        self.model.train(
            self.data_to_input(X), 
            self.data_to_output(X, Y), 
            **kwargs)

    def data_to_input(self, X):
        X = X - X.mean(dim=[1,2])
        X = (X - self.X_min) / (self.X_max - self.X_min)
        return X

    def data_to_output(self, X, Y):
        Y = Y - X.mean(dim=[1,2])
        Y = (Y - self.X_min) / (self.X_max - self.X_min)
        return Y

    def output_to_data(self, X, Y):
        Y = Y * (self.X_max - self.X_min) + self.X_min
        Y = Y + X.mean()
        return Y

    def forward(self, X, **kwargs):
        squeezed = len(X.shape) == 2
        if squeezed: X = X.unsqueeze(0)

        Y = self.data_to_input(X)
        Y = self.model(Y, **kwargs)
        Y = self.output_to_data(X, Y)

        if squeezed: Y = Y.squeeze(0)
        return Y