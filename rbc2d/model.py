import torch

# BC's for RBC
# Velocity:
# for i = 0,1, j=0,-1:
# - X[:,i,j,:] = 0 (bottom, top, Dirichlet)
# - X[:,2,0,:] = 1 (bottom, Dirichlet)
# - X[:,2,-1,:] = 0 (top, Dirichlet)
# - X[:,:,:,0] = X[:,:,:,-1] (left, right, periodic)
class RBC2DModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__initialized = False
        self.model = model

    def initialize(self, X, Y):
        n_channels = X.shape[1]
        self.X_min = torch.tensor([ X[:,k,:,:].min() for k in range(n_channels) ]) # - 0.1
        self.X_max = torch.tensor([ X[:,k,:,:].max() for k in range(n_channels) ]) # + 0.1
        for k in range(n_channels): 
            if self.X_max[k] == self.X_min[k]:
                self.X_max[k] = 1
        self.X_min = self.X_min.view(1,n_channels,1,1)
        self.X_max = self.X_max.view(1,n_channels,1,1)
        self.__initialized = True

    def train(self, X, Y, **kwargs):
        self.model.train(
            self.data_to_input(X), 
            self.data_to_output(X, Y), 
            **kwargs)

    def data_to_input(self, X):
        # return X
        X = (X - self.X_min) / (self.X_max - self.X_min)
        return X

    def data_to_output(self, X, Y):
        # return Y
        Y = (Y - self.X_min) / (self.X_max - self.X_min)
        return Y

    def output_to_data(self, X, Y):
        # return Y
        Y = Y * (self.X_max - self.X_min) + self.X_min
        return Y

    def forward(self, X, normalize_input=True):
        assert self.__initialized
        squeezed = len(X.shape) == 3
        if squeezed: X = X.unsqueeze(0)

        if normalize_input: Y = self.data_to_input(X)
        else: Y = X

        Y = self.model(Y)
        Y = self.output_to_data(X, Y)

        if squeezed: Y = Y.squeeze(0)
        return Y

class ELM2D(torch.nn.Module):
    def __init__(self, extent, step, in_channels, hidden, out_channels):

        super().__init__()
        self.hidden = hidden
        self.step = torch.tensor(step)
        self.extent = torch.tensor(extent)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = self.step+2*self.extent

        self.nl = torch.nn.Softplus(beta=1)
        self.lin = torch.nn.Linear(self.in_channels * self.ks.prod().item(), self.hidden, dtype=torch.float64, bias=True)
        # self.lhidden = torch.nn.Linear(self.hidden, self.hidden, dtype=torch.float64, bias=True)
        self.lout = torch.nn.Linear(self.hidden, self.out_channels * self.step.prod().item(), dtype=torch.float64, bias=False)
        
    def phi(self, X):
        # return self.nl(self.lhidden(self.nl(self.lin(X))))
        return self.nl(self.lin(X))
        
    def pad(self, X):
        n_samples, n_channels, n_x, n_y = X.shape

        # adds periodic boundary conditions
        X_padded = torch.cat((X[:,:,:, -self.extent[1]:], X, X[:,:,:, :self.extent[1]]), dim=3)
        # X_padded = torch.cat((X_padded[:,:,-self.extent[0]:,:], X_padded, X_padded[:,:,:self.extent[0],:]), dim=2)
        # return X_padded

        # adds constant terms
        n_y_padded = n_y + 2*self.extent[1]
        U_bc = torch.zeros((n_samples, 2, self.extent[0], n_y_padded))
        T_bc_top = 1*torch.ones((n_samples, 1, self.extent[0], n_y_padded)) # -1
        T_bc_btm = 2*torch.ones((n_samples, 1, self.extent[0], n_y_padded)) #  0
        P_bc_top = X_padded[:,None,3,None,-1,:].expand(-1,-1,self.extent[1],-1)
        P_bc_btm = X_padded[:,None,3,None, 0,:].expand(-1,-1,self.extent[1],-1)

        # b_bc_top = torch.ones_like(P_bc_top)
        # b_bc_btm = torch.ones_like(P_bc_btm)

        b_bc_top = torch.zeros_like(P_bc_top)
        b_bc_btm = torch.zeros_like(P_bc_btm)

        bc_top = torch.cat((U_bc, T_bc_top, P_bc_top, b_bc_top), dim=1)
        bc_btm = torch.cat((U_bc, T_bc_btm, P_bc_btm, b_bc_btm), dim=1)


        X_padded = torch.cat((bc_btm, X_padded, bc_top), dim=2)
        return X_padded

    def batched(self, X):
        # X = X.unsqueeze(1)
        X = torch.nn.functional.unfold(X, self.ks, stride=self.step)
        X = X.permute((0,2,1))
        return X
    
    def unbatch(self, X, output_dim):
        X = X.permute((0,2,1))
        X = torch.nn.functional.fold(X, output_dim, self.step, stride=self.step)
        # X = X.squeeze(1)
        return X

    def forward(self, X):
        D = torch.tensor(X.shape[2:])
        assert (D % self.step == 0).all()

        X = self.pad(X)
        X = self.batched(X)
        X = self.lout(self.phi(X))
        X = self.unbatch(X, D)

        return X

    def train(self, X, Y, batch=1000, noise=0, logging=True):
        bs, _, N0, N1 = X.shape

        X_train = torch.zeros(batch, self.in_channels,  self.ks[0], self.ks[1], dtype=torch.double)
        Y_train = torch.zeros(batch, self.out_channels, self.step[0], self.step[1], dtype=torch.double)
        
        for it in range(batch): 
            # selects a random position
            i = torch.randint(bs,(1,))[0]
            j0 = torch.randint(N0-self.ks[0],(1,))[0]
            j1 = torch.randint(N1-self.ks[1],(1,))[0]

            # selects the appropriate windows
            X_train[it] = X[i,   :, j0:j0+self.ks[0],
                                    j1:j1+self.ks[1]]
            Y_train[it] = Y[i,   :, j0+self.extent[0]:j0+self.ks[0]-self.extent[0],
                                    j1+self.extent[1]:j1+self.ks[1]-self.extent[1]]
            if logging: print(f"{it} / {batch}", end="\r")

        X_train = X_train.reshape(-1,self.in_channels *self.ks[0]  *self.ks[1]  )
        Y_train = Y_train.reshape(-1,self.out_channels*self.step[0]*self.step[1])

        if noise > 0:
            noise_distr = torch.distributions.Normal(0,noise)
            X_train = X_train + noise_distr.sample(X_train.shape)

        PhiX = self.phi(X_train).detach()
        # A = torch.linalg.lstsq(PhiX.cpu(), Y_train.cpu(), driver="gelsd").solution.cuda()
        A = torch.linalg.lstsq(PhiX, Y_train).solution
        self.lout.weight.data = A.transpose(0,1)