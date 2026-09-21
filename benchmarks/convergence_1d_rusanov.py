import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

C = 8
XMIN,XMAX = -30,30
SIGMA = 1
FINAL_TIME = 1.0
CFL = 0.5

NX_VALUES = [250,500,1000,2000,4000,8000]


def initial_condition(x,dx):
    u = np.exp(-x**2/(2*SIGMA**2))
    return u/(np.sum(u)*dx)


def dudt(u,dx):
    u_left = u
    u_right = np.roll(u,-1)

    flux = (
        0.5*C*(u_left+u_right)
        - 0.5*abs(C)*(u_right-u_left)
    )

    return -(flux-np.roll(flux,1))/dx


def tvd_rk3_step(u,dx,dt):

    k1 = dudt(u,dx)
    u1 = u+dt*k1

    k2 = dudt(u1,dx)
    u2 = 0.75*u+0.25*(u1+dt*k2)

    k3 = dudt(u2,dx)

    return (1/3)*u+(2/3)*(u2+dt*k3)


def analytical_solution(x,t):

    u = np.exp(-(x-C*t)**2/(2*SIGMA**2))
    return u/(np.sum(u)*(x[1]-x[0]))


errors = []
dx_values = []

for NX in NX_VALUES:

    x = np.linspace(XMIN,XMAX,NX,endpoint=False)
    dx = x[1]-x[0]

    dt = CFL*dx/abs(C)

    u = initial_condition(x,dx)

    t = 0

    while t < FINAL_TIME-1e-12:

        current_dt = min(dt,FINAL_TIME-t)

        u = tvd_rk3_step(u,dx,current_dt)

        t += current_dt

    analytical = analytical_solution(x,FINAL_TIME)

    error = np.linalg.norm(u-analytical)/np.linalg.norm(analytical)

    dx_values.append(dx)
    errors.append(error)

    print(f"NX = {NX}")
    print(f"dx = {dx:.6e}")
    print(f"Relative Error = {error:.6e}")
    print()


dx_values = np.array(dx_values)
errors = np.array(errors)

orders = (
    np.log(errors[:-1]/errors[1:])
    / np.log(dx_values[:-1]/dx_values[1:])
)

print("Observed convergence orders:")

for i,order in enumerate(orders):
    print(f"{NX_VALUES[i]} -> {NX_VALUES[i+1]} : {order:.4f}")


plt.figure(figsize=(8,6))

plt.loglog(dx_values,errors,'o-',label="Numerical error")

reference = errors[0]*(dx_values/dx_values[0])

plt.loglog(
    dx_values,
    reference,
    '--',
    label="First-order reference"
)

plt.xlabel("dx")
plt.ylabel("Relative L2 Error")
plt.title("Convergence: 1D Rusanov Advection")

plt.grid(True,which="both")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "convergence_1d_rusanov.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()