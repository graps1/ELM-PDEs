import torch

class KS1DModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__is_trained = False
        self.model = model

    def train(self, X, Y, **kwargs):
        mean = X.mean(dim=[1])
        self.X_min = (X - mean[:,None]).min()
        self.X_max = (X - mean[:,None]).max()

        self.model.train(
            self.data_to_input(X), 
            self.data_to_output(X, Y), 
            **kwargs)

        self.__is_trained = True

    def data_to_input(self, X):
        X = (X - self.X_min) / (self.X_max - self.X_min)
        return X

    def data_to_output(self, X, Y):
        Y = (Y - self.X_min) / (self.X_max - self.X_min)
        return Y

    def output_to_data(self, X, Y):
        Y = Y * (self.X_max - self.X_min) + self.X_min
        return Y

    def forward(self, X):
        assert self.__is_trained
        squeezed = len(X.shape) == 1
        if squeezed: X = X.unsqueeze(0)

        Y = self.data_to_input(X)
        Y = self.model(Y)
        Y = self.output_to_data(X, Y)

        if squeezed: Y = Y.squeeze(0)
        return Y