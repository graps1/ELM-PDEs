import torch

class ELM1D(torch.nn.Module):
    def __init__(self, extent, step, hidden, positions=0):
        super().__init__()
        self.hidden = hidden
        self.ks = step+2*extent
        self.step = step
        self.extent = extent
        self.positions = positions
        self.in_features = self.ks+self.positions
        self.lin = torch.nn.Linear(self.in_features, self.hidden, dtype=torch.float64, bias=True)
        self.nl = torch.nn.Softplus()
        self.lout = torch.nn.Linear(self.hidden, self.step, dtype=torch.float64, bias=False)
        
    def phi(self, X):
        return self.nl(self.lin(X))

    def positional_encoding(self, N):
        lin = torch.linspace(0, 1, N // self.step) + (self.step / N) / 2
        pos = [ torch.cos(2**k*torch.pi*lin) for k in range(self.positions)]
        pos = torch.stack(pos, dim=1)
        return pos

    def pad(self, X):
        return torch.nn.functional.pad(X, (self.extent,self.extent), "circular")

    def batched(self, X):
        return X.unfold(1,self.ks,self.step)

    def forward(self, X):
        M, N = X.shape
        assert N % self.step == 0

        X = self.pad(X)
        X = self.batched(X)

        if self.positions>0: 
            pos = self.positional_encoding(N)
            pos = pos.expand(M, -1, -1)
            X = torch.cat((X,pos),dim=2)

        X = self.lout(self.phi(X))
        X = X.view(M, -1)
        return X
    
    def train(self, X, Y, stopping_threshold=1e-5, noise=1e-4, logging=True, callback=None):
        if noise > 0: noise_distr = torch.distributions.Normal(0,noise)

        M, N = X.shape
        E_PhiTPhi = 0
        E_PhiTY = 0
        change = 1
        k = 0

        while change > stopping_threshold:
            i = torch.randint(M-1,(1,))[0]

            X_train = X[i,None,...]
            X_train = self.batched(self.pad(X_train))
            Y_train = Y[i,None,...]
            Y_train = self.batched(self.pad(Y_train))
            Y_train = Y_train[:, :, self.extent:self.ks-self.extent]

            if self.positions > 0:
                pos = self.positional_encoding(N)
                pos = pos.unsqueeze(0)
                X_train = torch.cat((X_train,pos),dim=2)
            
            j = torch.randint(X_train.shape[1],(1,))[0]
            X_train = X_train[:,j,:]
            Y_train = Y_train[:,j,:]
            if noise > 0: X_train += noise_distr.sample(X_train.shape) 

            Phi = self.phi(X_train).detach()

            alpha = 1/(k+1)
            E_PhiTPhi_ =  (1-alpha)*E_PhiTPhi  + alpha*(Phi.T @ Phi)
            E_PhiTY_   =  (1-alpha)*E_PhiTY    + alpha*(Phi.T @ Y_train)
            change = 0.99*change + 0.01*((E_PhiTPhi_ - E_PhiTPhi).pow(2).sum() + 
                                            (E_PhiTY_   - E_PhiTY  ).pow(2).sum())
            if logging: print(f"change = {change:.6f}", end="\r")
            if change <= stopping_threshold: break
            E_PhiTPhi, E_PhiTY = E_PhiTPhi_, E_PhiTY_ 
            if callback is not None and k > 1: 
                A = torch.linalg.solve(E_PhiTPhi, E_PhiTY)
                self.lout.weight.data = A.transpose(0,1)
                callback(k, change, self)
            k += 1

        A = torch.linalg.solve(E_PhiTPhi, E_PhiTY)
        self.lout.weight.data = A.transpose(0,1)


class ELM2D(torch.nn.Module):
    def __init__(self, extent, step, hidden):
        super().__init__()
        self.hidden = hidden
        self.step = torch.tensor(step)
        self.extent = torch.tensor(extent)
        self.ks = self.step+2*self.extent
        self.lin = torch.nn.Linear(self.ks.prod().item(), self.hidden, dtype=torch.float64, bias=True)
        self.nl = torch.nn.Softplus()
        self.lout = torch.nn.Linear(self.hidden, self.step.prod().item(), dtype=torch.float64, bias=False)
        
    def phi(self, X):
        return self.nl(self.lin(X))
        
    def pad(self, X):
        return torch.nn.functional.pad(X, 
            (self.extent[0],self.extent[0],self.extent[1],self.extent[1]), 
            "circular")

    def batched(self, X):
        X = X.unsqueeze(1)
        X = torch.nn.functional.unfold(X, self.ks, stride=self.step)
        X = X.permute((0,2,1))
        return X
    
    def unbatch(self, X, output_dim):
        X = X.permute((0,2,1))
        X = torch.nn.functional.fold(X, output_dim, self.step, stride=self.step)
        X = X.squeeze(1)
        return X

    def forward(self, X, nr_symmetries=1):
        D = torch.tensor(X.shape[1:])
        assert (D % self.step == 0).all()

        X = apply_symmetries(X, nr_symmetries)
        X = X.view(-1, *X.shape[2:])

        X = self.pad(X)
        X = self.batched(X)
        X = self.lout(self.phi(X))
        X = self.unbatch(X, D)

        X = X.view(-1, nr_symmetries, *X.shape[1:])
        X = apply_symmetries(X, nr_symmetries, inv=True)
        X = X.mean(dim=1)

        return X

    def train(self, X, Y, nr_symmetries=1, stopping_threshold=1e-5, noise=1e-4, logging=True, callback=None):
        bs, N0, N1 = X.shape
        X = apply_symmetries(X, nr_symmetries=nr_symmetries)
        Y = apply_symmetries(Y, nr_symmetries=nr_symmetries)

        if noise > 0: noise_distr = torch.distributions.Normal(0,noise)

        E_PhiTPhi = 0
        E_PhiTY = 0
        change = None
        k = 0

        while change is None or change > stopping_threshold:
            # selects a random position
            i = torch.randint(bs,(1,))[0]
            j0 = torch.randint(N0-self.ks[0],(1,))[0]
            j1 = torch.randint(N1-self.ks[1],(1,))[0]

            # selects the appropriate windows
            X_train = X[i,   :, j0:j0+self.ks[0],
                                j1:j1+self.ks[1]]
            Y_train = Y[i,   :, j0+self.extent[0]:j0+self.ks[0]-self.extent[0],
                                j1+self.extent[1]:j1+self.ks[1]-self.extent[1]]
            
            X_train = X_train.reshape(-1,self.ks[0]*self.ks[1])
            Y_train = Y_train.reshape(-1,self.step[0]*self.step[1])

            # adds noise -> increases robustness
            if noise > 0: X_train += noise_distr.sample(X_train.shape) 
            Phi = self.phi(X_train).detach()

            alpha = 1/(k+1)
            E_PhiTPhi_ =  (1-alpha)*E_PhiTPhi  + alpha*(Phi.T @ Phi)/nr_symmetries
            E_PhiTY_   =  (1-alpha)*E_PhiTY    + alpha*(Phi.T @ Y_train)/nr_symmetries
            target = ((E_PhiTPhi_-E_PhiTPhi).pow(2).sum() + (E_PhiTY_-E_PhiTY  ).pow(2).sum())
            if change is None: change = target
            else: change = 0.95*change + 0.05*target
            if logging: print(f"change = {change:.6f}", end="\r")
            if change < stopping_threshold: break
            E_PhiTPhi, E_PhiTY = E_PhiTPhi_, E_PhiTY_ 
            if callback is not None and k > 1: 
                A = torch.linalg.solve(E_PhiTPhi, E_PhiTY)
                self.lout.weight.data = A.transpose(0,1)
                callback(k, change, self)
            k += 1

        A = torch.linalg.solve(E_PhiTPhi, E_PhiTY)
        self.lout.weight.data = A.transpose(0,1)

def apply_symmetries(X, nr_symmetries, inv=False):
    assert len(X.shape) in [3,4]

    # expand tensor
    if len(X.shape) == 3: 
        X = X.unsqueeze(1)
        X = X.repeat(1,nr_symmetries,1,1)

    if nr_symmetries >= 2: X[:,1,:,:] = X[:,1,:,:].clone().flip([1])
    if nr_symmetries >= 3: X[:,2,:,:] = X[:,2,:,:].clone().flip([2])
    if nr_symmetries >= 4: X[:,3,:,:] = X[:,3,:,:].clone().flip([1,2])
    if nr_symmetries >= 5: X[:,4,:,:] = X[:,4,:,:].clone().transpose(1,2)
    if inv:
        if nr_symmetries >= 6: X[:,5,:,:] = X[:,5,:,:].clone().transpose(1,2).flip([1])
        if nr_symmetries >= 7: X[:,6,:,:] = X[:,6,:,:].clone().transpose(1,2).flip([2])
        if nr_symmetries >= 8: X[:,7,:,:] = X[:,7,:,:].clone().transpose(1,2).flip([1,2])
    else:
        if nr_symmetries >= 6: X[:,5,:,:] = X[:,5,:,:].clone().flip([1]).transpose(1,2)
        if nr_symmetries >= 7: X[:,6,:,:] = X[:,6,:,:].clone().flip([2]).transpose(1,2)
        if nr_symmetries >= 8: X[:,7,:,:] = X[:,7,:,:].clone().flip([1,2]).transpose(1,2)
    return X
