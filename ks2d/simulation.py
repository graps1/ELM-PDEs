from sympy import symbols, exp, pi
import numpy as np
from shenfun import *

dt = 0.01
save_period_long = 50 # dt = 0.01s, period = 50 => saves one frame all 0.5s
start_time = 200
end_time_long = start_time + 30
end_time_short = start_time + 2*dt
batch_long = []
batch_short = []
f_short = open('ks2d_short.npy', 'wb')
f_long = open('ks2d_long.npy', 'wb')

# ---SHENFUN STUFF---

# Use sympy to set up initial condition
x, y = symbols("x,y", real=True)
ue = exp(-0.01*((x-30*pi)**2+(y-30*pi)**2))        # + exp(-0.02*((x-15*np.pi)**2+(y)**2))

# Size of discretization
N = (256, 256)

K0 = FunctionSpace(N[0], 'F', dtype='D', domain=(0, 60*np.pi))
K1 = FunctionSpace(N[1], 'F', dtype='d', domain=(0, 60*np.pi))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})
TV = VectorSpace(T)
padding_factor = 1.5
Tp = T.get_dealiased()
TVp = VectorSpace(Tp)
gradu = Array(TVp)

u = TrialFunction(T)
v = TestFunction(T)

# Create solution and work arrays
U = Array(T, buffer=ue)
U_hat = Function(T)
gradu = Array(TVp)
K = np.array(T.local_wavenumbers(True, True, True))
mask = T.get_mask_nyquist()

def LinearRHS(self, u, **params):
    # Assemble diagonal bilinear forms
    return -div(grad(u))-div(grad(div(grad(u))))

def NonlinearRHS(self, U, U_hat, dU, gradu, **params):
    # Assemble nonlinear term
    gradu = TVp.backward(1j*K*U_hat, gradu)
    dU = Tp.forward(0.5*(gradu[0]*gradu[0]+gradu[1]*gradu[1]), dU)
    dU.mask_nyquist(mask)
    if comm.Get_rank() == 0:
        dU[0, 0] = 0
    return -dU

#initialize
X = T.local_mesh(True)
U_hat = U.forward(U_hat)
U_hat.mask_nyquist(mask)

# Integrate using an exponential time integrator
def update(self, u, u_hat, t, tstep, **params):
    print(f"time = {t:.3f}/{end_time_long:.3f}",end="\r")
    if round(t/dt) % save_period_long == 0 and start_time <= t:
        u = u_hat.backward(u)
        batch_long.append(np.array(u)) # simply add coefficients into array
    if start_time <= t and t <= end_time_short:
        u = u_hat.backward(u)
        batch_short.append(np.array(u)) # simply add coefficients into array

if __name__ == '__main__':
    par = {
           'gradu': gradu,
           'count': 0}
    print("starting simulation ...")

    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    #integrator = RK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    U_hat = integrator.solve(U, U_hat, dt, (0, end_time_long))
    np.save(f_short, np.array(batch_short))
    np.save(f_long, np.array(batch_long))
    f_short.close()
    f_long.close()