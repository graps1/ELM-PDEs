# from sympy import symbols, exp, pi
import numpy as np
from shenfun import *
from math import remainder

np.random.seed(0)

dt = 0.05
gamma = 0.5

# save_period_very_short = 0.001
save_period_short = 0.05
save_period_long = 0.5

# end_time_very_short = 10*save_period_very_short
start_time_short = 90*save_period_short
end_time_short = 110*save_period_short
end_time_long = 20

# batch_very_short = []
batch_short = []
batch_long = []

# f_very_short = open('ks2d_very_short.npy', 'wb')
f_short = open('ch2d_short.npy', 'wb')
f_long = open('ch2d_long.npy', 'wb')

# ---SHENFUN STUFF---

# Size of discretization
N = (512, 512)

K0 = FunctionSpace(N[0], 'F', dtype='D', domain=(0, 100))
K1 = FunctionSpace(N[1], 'F', dtype='d', domain=(0, 100))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})
TV = VectorSpace(T)
gradu = Array(TV)

u = TrialFunction(T)
v = TestFunction(T)

# Create solution and work arrays
U = Array(T, buffer=np.random.uniform(-1,1,size=N))
U_hat = Function(T)
gradu = Array(TV)
K = np.array(T.local_wavenumbers(True, True, True))
mask = T.get_mask_nyquist()

K2 = np.sum(K**2, 0)
K4 = K2**2
# work = Function(T)
# rhs_divgrad = Inner(v, div(grad(work)))

def NonlinearRHS(self, U, U_hat, dU, gradu, **params):
    # global work
    Ub = U_hat.backward() # padding_factor=1.5)
    Ub[:] = Ub**3
    dU[:] = -K2 * Ub.forward()
    # work = Ub.forward(work)
    # dU[:] = -K2*work
    # dU[:] = rhs_divgrad()
    return dU

def LinearRHS(self, u, **params):
    global gamma
    # return -div(grad(u)) - gamma*div(grad(div(grad(u))))
    return K2 - gamma*K4

# def LinearRHS(self, u, **params):
#     global gamma
#     return -div(grad(u))-gamma*div(grad(div(grad(u))))
# 
# def NonlinearRHS(self, U, U_hat, dU, gradu, **params):
#     dU = (div(grad(U**3))).forward(dU)
#     dU.mask_nyquist(mask)
#     return dU

#initialize
X = T.local_mesh(True)
U_hat = U.forward(U_hat)
U_hat.mask_nyquist(mask)

# Integrate using an exponential time integrator
def update(self, u, u_hat, t, tstep, **params):
    print(f"time = {t:.3f}/{end_time_long:.3f}",end="\r")
    u = u_hat.backward(u)
    if abs(remainder(t, save_period_long)) < 1e-8:
        batch_long.append(np.array(u))
    if abs(remainder(t, save_period_short)) < 1e-8 and t <= end_time_short and start_time_short <= t:
        batch_short.append(np.array(u))

if __name__ == '__main__':
    par = {
           'gradu': gradu,
           'count': 0}
    print("starting simulation ...")

    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    U_hat = integrator.solve(U, U_hat, dt, (0, end_time_long))
    np.save(f_short, np.array(batch_short))
    np.save(f_long, np.array(batch_long))
    f_short.close()
    f_long.close()