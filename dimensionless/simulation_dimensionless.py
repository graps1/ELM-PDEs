import shenfun as sf
import numpy as np

np.random.seed(0)

dt = 0.01
start_time = 0
end_time = 100
N = 256
u0 = np.random.uniform(-.4, .4, N)

# SHENFUN stuff

T = sf.FunctionSpace(N, 'F', dtype='d', domain=(0, 60*np.pi))
v = sf.TestFunction(T)
u_ = sf.Array(T)
u_x = sf.Array(T)
u_hat = sf.Function(T)
f = open('ks1d_dimensionless.npy', 'wb')
k = T.wavenumbers(scaled=True, eliminate_highest_freq=True)

batch = []

def LinearRHS(self, u, **params):
    A = sf.inner(v, -sf.Dx(u, 0, 4)-sf.Dx(u, 0, 2)) # Two matrices in list A
    A[0] += A[1] # Add second matrix to the first
    return A[0]  # return just the first (which is now the sum of both)

def NonlinearRHS(_0, _1, u_hat, _2, **params):
    u_x[:] = (1j * k * u_hat).backward() # compute coefficients of u_hat and put them into u_
    rhs =  T.forward(0.5*u_x*u_x) 
    return -rhs

def update(self, u, u_hat, t, tstep, **params):
    u = u_hat.backward(u)
    print(f"{t:.3f}/{end_time:.3f}", end="\r")
    if t >= start_time: batch.append(np.array(u)) 

integrator = sf.ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update)

if __name__ == "__main__":
    print("starting simulation ...")

    u_[:] = u0
    u_hat = u_.forward(u_hat)
    integrator.setup(dt)
    u_hat = integrator.solve(u_, u_hat, dt, (0, end_time))
    np.save(f, np.array(batch))
    batch.clear()
    f.close()