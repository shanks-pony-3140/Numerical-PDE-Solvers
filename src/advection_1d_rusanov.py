import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 1D linear advection: u_t + c u_x = 0
C=8.0
XMIN,XMAX= -30.0,30.0
NX= 9000
CFL= 0.5
SIGMA= 1.0
OUTPUT_TIMES= [0.1,0.3,1,3]

x= np.linspace(XMIN,XMAX,NX,endpoint=False)
dx= x[1]-x[0]
dt= CFL*dx/abs(C)


def initial_condition(x):
    u= np.exp(-x**2/(2*SIGMA**2))
    return u/(np.sum(u)*dx)


def dudt(u):
    # Rusanov numerical flux with pbc
    u_left= u
    u_right= np.roll(u, -1)
    flux= 0.5*C* (u_left+u_right) - 0.5*abs(C)*(u_right-u_left)
    return -(flux-np.roll(flux,1))/dx


def tvd_rk3_step(u):
    k1= dudt(u)
    u1= u+dt*k1
    k2= dudt(u1)
    u2= 0.75*u+0.25*(u1+dt*k2)
    k3 = dudt(u2)
    
    return (1/3)*u+(2/3)*(u2+dt*k3)


def analytical_solution(x, t):
    u = np.exp(-(x-C*t)**2/(2*SIGMA**2))
    return u/(np.sum(u)*dx)


def errors(numerical,analytical):
    
    mass_numerical= np.sum(numerical)*dx
    mass_analytical= np.sum(analytical)*dx
    max_error= np.max(np.abs(numerical-analytical))
    l2_error= np.sqrt(np.mean((numerical-analytical)**2))
    relative_error= np.linalg.norm(numerical-analytical)/np.linalg.norm(analytical)*100
    
    return mass_numerical,mass_analytical,max_error,l2_error,relative_error



u= initial_condition(x)
snapshots= {}
t= 0

for target_time in OUTPUT_TIMES:
    while t<target_time-1e-12:
        u= tvd_rk3_step(u)
        t+= dt
    snapshots[target_time]= u.copy()

for t,numerical in snapshots.items():
    analytical= analytical_solution(x,t)
    mass_n,mass_a,max_error,l2_error,relative_error= errors(numerical,analytical)

    print(f"t = {t}")
    print(f"Numerical Mass  = {mass_n:.10f}")
    print(f"Analytical Mass = {mass_a:.10f}")
    print(f"Max Error       = {max_error:.3e}")
    print(f"L2 Error        = {l2_error:.3e}")
    print(f"Relative Error  = {relative_error:.6f}%")
    print()

    plt.figure(figsize=(10, 5))
    plt.plot(x,numerical,label="Rusanov + TVD-RK3")
    plt.plot(x,analytical, "--", label="Analytical")
    plt.xlim(XMIN, XMAX)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"1D Advection: t = {t}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"advection_1d_rusanov_t{t:g}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()



