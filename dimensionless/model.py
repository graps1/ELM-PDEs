import torch

def pad(X, axis, extent, mode="circular", val_left=None, val_right=None):
    X = X.transpose(axis, -1)

    if mode == "circular":
        assert extent <= X.shape[-1]
        left, right = X[...,-extent:], X[...,:extent]
    elif mode == "dirichlet": 
        shape = list(X.shape[:-1]) + [extent]
        left, right = torch.ones(shape)*val_left, torch.ones(shape)*val_right
    else:
        raise Exception

    X = torch.cat((left, X, right), dim=-1)
    X = X.transpose(axis, -1)
    return X

class PointwiseLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = torch.nn.Linear(in_dim, out_dim, bias=False).double()

    def forward(self, X):
        X = X.transpose(1,-1) # shifts the channel axis to the last index
        Y = self.layer(X)
        Y = Y.transpose(1,-1)
        return Y

    def train(self, X, Y):
        X = X.transpose(1,-1) # shifts the channel axis to the last index
        X = X.reshape(-1,X.shape[-1]) # conflates the first few axes, except for input dimension
        Y = Y.transpose(1,-1).reshape(-1, 1) # shifts the channel axis to the last index
        A = torch.linalg.lstsq(X.cpu(), Y.cpu(), driver="gelsd", rcond=1e-8).solution.transpose(0,1)
        self.layer.weight.data = A.to(torch.get_default_device())

class FiniteDifferenceLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # hard-coded finite-difference kernels
        # dx = torch.tensor([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280 ]).double()
        # dxx = torch.tensor([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560 ]).double()
        # dxxxx = torch.tensor([7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240 ]).double()
        # dxx = torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12]).double()
        # dxx = torch.tensor([1, -2, 1]).double()

        self.extent = 1
        self.ks = 2*self.extent + 1 # dxx.shape[0]
        self.nr_kernels = 2
        # self.conv_dx = torch.nn.Conv1d(1, 1, self.ks, bias=False).double()
        # self.conv_dx.weight.data = dx.view(1,1,self.ks)
        # self.conv_dxx = torch.nn.Conv1d(1, 1, self.ks, bias=False).double()

        self.conv1d = torch.nn.Conv1d(self.nr_kernels, self.nr_kernels, self.ks, groups=self.nr_kernels, bias=False).double()
        self.conv2d = torch.nn.Conv2d(self.nr_kernels, self.nr_kernels, self.ks, groups=self.nr_kernels, bias=False).double()

        # print( self.conv2d.weight.data.shape )
        # self.conv2d.weight.data = make_invariant_kernels(self.channels, self.ks).unsqueeze(1).double()

        # laplace1d = torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12]).double()
        laplace1d = torch.tensor([1, -2, 1]).double()
        identity1d = torch.tensor([0,1,0]).double()
        self.conv1d.weight.data[0,0,:] = laplace1d
        self.conv1d.weight.data[1,0,:] = identity1d

        # laplace2d = torch.tensor([[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],[0,0,1,0,0],[0,0,0,0,0]]).double()
        # laplace2d = torch.tensor([
        #     [ 0.048,  0.328,  0.436,  0.328,  0.048],
        #     [ 0.328, -0.3  , -0.54 , -0.3  ,  0.328],
        #     [ 0.436, -0.54 , -1.2  , -0.54 ,  0.436],
        #     [ 0.328, -0.3  , -0.54 , -0.3  ,  0.328],
        #     [ 0.048,  0.328,  0.436,  0.328,  0.048]])
        # identity2d = torch.tensor([
        #     [0,0,0,0,0],
        #     [0,0,0,0,0],
        #     [0,0,1,0,0],
        #     [0,0,0,0,0],
        #     [0,0,0,0,0]]).double()
        # laplace2d = torch.tensor([
        #     [ 0.107,  0.241,  0.107],
        #     [ 0.241, -1.39,   0.241],
        #     [ 0.107,  0.241,  0.107]])
        laplace2d = torch.tensor([[1,2,1],[2,-12,2],[1,2,1]]).double() / 4
        identity2d = torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).double()
        self.conv2d.weight.data[0,0,:,:] = laplace2d
        self.conv2d.weight.data[1,0,:,:] = identity2d

    def forward(self, X):
        # X has shape: batch x channels x dim1 x dim2 x ...
        # computes a vector of size batch x channels x dim1 x dim2 x ...
        dims_shape = X.shape[2:]
        nr_dims = len(dims_shape)
        new_dims_shape = torch.tensor(dims_shape) - 2*self.extent

        assert nr_dims in [1,2], "other dimension sizes not yet implemented"

        orig_channels = X.shape[1]
        assert orig_channels % self.nr_kernels == 0

        X = X.view(-1, self.nr_kernels, *dims_shape)
        if nr_dims == 1: X = self.conv1d(X)
        if nr_dims == 2: X = self.conv2d(X)
        X = X.view(-1, orig_channels, *new_dims_shape)
        return X
    
class Model(torch.nn.Module):
    def __init__(self, hidden1=16, hidden2=32, bc=None):
        super().__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.bc = bc

        self.nl = torch.nn.Softplus()
        self.block1 = torch.nn.Sequential(
            FiniteDifferenceLayer(),
            PointwiseLinear(2, self.hidden1),
            self.nl
        )
        self.block2 = torch.nn.Sequential(
            FiniteDifferenceLayer(),
            PointwiseLinear(self.hidden1, self.hidden2),
            self.nl
        )

        self.lin_out = PointwiseLinear(self.hidden2, 1)
        self.extent = self.block1[0].extent + self.block2[0].extent

    def pad(self, X):
        for i in range(len(X.shape)-1): 
            X = pad(X, axis=i+1, extent=self.extent, **self.bc[i])
        return X

    def phi(self, X):
        # X has shape: batch x dim1 x dim2 x ...
        X = self.pad(X)
        X = X.unsqueeze(1)
        X = X.repeat(1, 2, *[1]*len(X.shape[2:]))
        X = self.block1(X)
        X = self.block2(X)
        return X

    def forward(self, X):
        dims = list(range(1,len(X.shape)))
        Y = X + self.lin_out(self.phi(X - X.mean(dim=dims))).squeeze(1)
        Y = X.mean(dims) + Y - Y.mean(dims)
        return Y

    def train(self, X, Y):
        dims = list(range(1,len(X.shape)))
        self.lin_out.train(self.phi(X - X.mean(dim=dims)), Y - X)