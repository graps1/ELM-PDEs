import torch
import numpy as np

class ELM1D(torch.nn.Module):
    def __init__(self, extent, step, hidden, positions=0):
        super().__init__()
        self.hidden = hidden
        self.ks = step+2*extent
        self.step = step
        self.extent = extent
        self.positions = positions
        self.in_features = self.ks+self.positions

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.hidden, dtype=torch.float64, bias=True),
            torch.nn.Softplus())
        self.lout = torch.nn.Linear(
            self.hidden, self.step, dtype=torch.float64, bias=False)

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
    
    def train(self, D, stopping_threshold=1e-5, noise=1e-4, logging=True, callback=None):
        if noise > 0: noise_distr = torch.distributions.Normal(0,noise)

        E_PhiTPhi = 0
        E_PhiTY = 0
        change = 1
        k = 0

        while change > stopping_threshold:
            for d in D:
                M, N = d.shape
                i = torch.randint(M-1,(1,))[0]

                X_train = d[i,None,...]
                X_train = self.batched(self.pad(X_train))
                Y_train = d[i+1,None,...]
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
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.ks.prod().item(), self.hidden, dtype=torch.float64, bias=True),
            torch.nn.Softplus())
        self.lout = torch.nn.Linear(
            self.hidden, self.step.prod().item(), dtype=torch.float64, bias=False)

    def pad(self, X):
        return torch.nn.functional.pad(X, 
            (self.extent[0],self.extent[0],self.extent[1],self.extent[1]), 
            "circular")

    def batched(self, X):
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
        X = X.unsqueeze(1)
        X = self.pad(X)
        X = self.batched(X)
        if nr_symmetries > 1:
            X = X.view(*X.shape[:2], self.ks[0], self.ks[1])
            X = expand_and_apply_symmetries(X, dims=[2,3], nr_symmetries=nr_symmetries)
            X = X.view(*X.shape[:3], self.ks[0]*self.ks[1])
        X = self.lout(self.phi(X))
        if nr_symmetries > 1:
            X = X.view(*X.shape[:3], self.step[0], self.step[1])
            X = apply_symmetries(X, dims=[2,3], nr_symmetries=nr_symmetries, inv=True)
            X = X.mean(dim=1)
            X = X.view(*X.shape[:2], self.step[0]*self.step[1])
        X = self.unbatch(X, D)
        return X
    
    def train(self, D, nr_symmetries=1, stopping_threshold=1e-5, noise=1e-4, logging=True, callback=None):
        if noise > 0: noise_distr = torch.distributions.Normal(0,noise)

        E_PhiTPhi = 0
        E_PhiTY = 0
        change = None
        k = 0

        while change is None or change > stopping_threshold:
            for d in D:
                # selects a random position
                N0, N1 = d.shape[1:]
                i = torch.randint(d.shape[0]-1,(1,))[0]
                j0 = torch.randint(N0-self.ks[0],(1,))[0]
                j1 = torch.randint(N1-self.ks[1],(1,))[0]

                # selects the appropriate windows
                X_train = d[i,   None, j0:j0+self.ks[0],
                                       j1:j1+self.ks[1]]
                Y_train = d[i+1, None, j0+self.extent[0]:j0+self.ks[0]-self.extent[0],
                                       j1+self.extent[1]:j1+self.ks[1]-self.extent[1]]

                if nr_symmetries > 1:
                    X_train = expand_and_apply_symmetries(X_train, nr_symmetries=nr_symmetries)
                    Y_train = expand_and_apply_symmetries(Y_train, nr_symmetries=nr_symmetries)
                
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


def expand_and_apply_symmetries(X, dims=[1,2], nr_symmetries=8, inv=False):
    X = X.unsqueeze(1)
    X = X.expand(X.shape[0],nr_symmetries,*X.shape[2:])
    return apply_symmetries(X, dims, nr_symmetries=nr_symmetries, inv=inv)

def apply_symmetries(X, dims=[1,2], nr_symmetries=8, inv=False):
    X_sym = [ R(X[:,g,...], g, dims, inv=inv) for g in range(nr_symmetries) ]
    return torch.stack(X_sym, dim=1)

def R(Y, g, dims, inv=False):
    d1, d2 = dims
    operations = [
        lambda x: x,
        lambda x: torch.flip(x, [d1]),
        lambda x: torch.flip(x, [d2]),
        lambda x: torch.flip(x, [d1,d2]),
        lambda x: x.transpose(d1,d2),
        (lambda x: torch.flip(x.transpose(d1,d2), [d1]))   if inv else 
        (lambda x: torch.flip(x, [d1]).transpose(d1,d2)),
        (lambda x: torch.flip(x.transpose(d1,d2), [d2]))   if inv else 
        (lambda x: torch.flip(x, [d2]).transpose(d1,d2)),
        (lambda x: torch.flip(x.transpose(d1,d2), [d1,d2])) if inv else 
        (lambda x: torch.flip(x, [d1,d2]).transpose(d1,d2))
    ]
    return operations[g](Y)


class Dataset:
    def __init__(self, fname):
        self.val = None
        self.f = None
        self.fname = fname

    def __del__(self):
        if self.f is not None: self.f.close()

    def __iter__(self):
        if self.f is not None: self.f.close()
        self.f = open(self.fname, 'rb')
        return self

    def __next__(self):
        try: 
            val = torch.tensor(np.load(self.f))
            return val
        except: 
            self.f = open(self.fname, 'rb')
            raise StopIteration