import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent/"figures"
FIGURES_DIR.mkdir(exist_ok=True)

C= 8
XMIN,XMAX= -30,30
SIGMA= 1
FINAL_TIME= 1.0

NX_VALUES= [250,500,1000,2000,4000]


def initial_condition(x,dx):
    u= np.exp(-x**2/(2*SIGMA**2))
    return u/(np.sum(u)*dx)


def dudt(u,dx):
    du = np.empty_like(u)

    du[1:-1]= -C*(u[2:]-u[:-2])/(2*dx)
    du[0]= -C*(u[1]-u[-1])/(2*dx)
    du[-1]= -C*(u[0]-u[-2])/(2*dx)

    return du


def rk4_step(u,dx,dt):

    k1= dudt(u,dx)
    k2= dudt(u+0.5*dt*k1,dx)
    k3= dudt(u+0.5*dt*k2,dx)
    k4= dudt(u+dt*k3,dx)

    return u + dt*(k1+2*k2+2*k3+k4)/6


def analytical_solution(x,t):

    u= np.exp(-(x-C*t)**2/(2*SIGMA**2))
    return u/(np.sum(u)*(x[1]-x[0]))


errors = []
dx_values = []

for NX in NX_VALUES:

    x= np.linspace(XMIN,XMAX,NX,endpoint=False)
    dx= x[1]-x[0]

    CFL= 0.2
    dt= CFL*dx/abs(C)

    u= initial_condition(x,dx)

    t= 0

    while t<FINAL_TIME-1e-12:

        current_dt= min(dt,FINAL_TIME-t)

        u= rk4_step(u,dx,current_dt)

        t+= current_dt

    analytical= analytical_solution(x,FINAL_TIME)

    error= np.linalg.norm(u-analytical)/np.linalg.norm(analytical)

    dx_values.append(dx)
    errors.append(error)

    print(f"NX = {NX}")
    print(f"dx = {dx:.6e}")
    print(f"Relative Error = {error:.6e}")
    print()


dx_values= np.array(dx_values)
errors= np.array(errors)

orders= np.log(errors[:-1]/errors[1:]) / np.log(dx_values[:-1]/dx_values[1:])

print("Observed convergence orders:")

for i,order in enumerate(orders):
    print(f"{NX_VALUES[i]} -> {NX_VALUES[i+1]} : {order:.4f}")


plt.figure(figsize=(8,6))

plt.loglog(dx_values,errors,'o-',label="Numerical error")

reference = errors[0]*(dx_values/dx_values[0])**2

plt.loglog(dx_values,reference,'--',label="Second-order reference")

plt.xlabel("dx")
plt.ylabel("Relative L2 Error")
plt.title("Convergence: 1D Central Advection")
plt.grid(True,which="both")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "convergence_1d_central.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()