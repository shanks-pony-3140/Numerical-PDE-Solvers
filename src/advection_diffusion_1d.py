import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 1D advection-diffusion: u_t+c u_x= D u_xx
C= 2
D= 2
XMIN,XMAX= -30,30
NX= 3000
DT= 1e-4
OUTPUT_TIMES= [0.1,0.3,1,3]

x= np.linspace(XMIN,XMAX,NX,endpoint=False)
dx= x[1]-x[0]


def initial_condition():
    #Discrete delta function
    u= np.zeros_like(x)
    u[NX//2]= 1.0/dx

    return u


def dudt(u):
    diffusion= D*(np.roll(u,-1)-2*u+np.roll(u,1))/dx**2
    advection= -C*(np.roll(u,-1)-np.roll(u,1))/(2*dx)

    return diffusion+advection


def rk4_step(u):

    k1= dudt(u)
    k2= dudt(u+0.5*DT*k1)
    k3= dudt(u+0.5*DT*k2)
    k4= dudt(u+DT*k3)

    return u + DT*(k1+2*k2+2*k3+k4)/6


def analytical_solution(t):
    return (1/np.sqrt(4*np.pi*D*t))*np.exp(-(x-C*t)**2/(4*D*t))


u=initial_condition()
snapshots={}
t=0
for target_time in OUTPUT_TIMES:
    while t < target_time-1e-12:
        u= rk4_step(u)
        t+=DT
    snapshots[target_time]=u.copy()
    
for t,numerical in snapshots.items():
    
    analytical= analytical_solution(t)
    
    mass_numerical= np.sum(numerical)*dx
    mass_analytical= np.sum(analytical)*dx
    max_error= np.max(np.abs(numerical-analytical))
    l2_error= np.sqrt(np.mean((numerical-analytical)**2))
    relative_error= np.linalg.norm(numerical-analytical)/np.linalg.norm(analytical)*100
    
    print(f"t= {t}")
    print(f"Numerical Mass  = {mass_numerical:.10f}")
    print(f"Analytical Mass = {mass_analytical:.10f}")
    print(f"Max Error       = {max_error:.3e}")
    print(f"L2 Error        = {l2_error:.3e}")
    print(f"Relative Error  = {relative_error:.6f}%")
    print()
    
    plt.figure(figsize=(10,5))
    
    plt.plot(x,numerical,label="Numerical")
    plt.plot(x,analytical,"--",label="Analytical")
    plt.xlim(XMIN,XMAX)
    plt.xlabel("x")
    plt.ylabel("u")
    
    plt.title(f"1D Advection-Diffusion: t= {t}")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"advection_diffusion_1d_t{t:g}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()



