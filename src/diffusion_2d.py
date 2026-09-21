import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 2D diffusion: u_t= Dx u_xx+Dy u_yy
DX= 2
DY= 2
XMIN,XMAX= -20,20
YMIN,YMAX= -20,20
NX= 200
NY= 200
DT= 5e-3
OUTPUT_TIMES= [0.1,0.3,1,3]

x= np.linspace(XMIN,XMAX,NX,endpoint=False)
y= np.linspace(YMIN,YMAX,NY,endpoint=False)
X,Y= np.meshgrid(x,y)
dx= x[1]-x[0]
dy= y[1]-y[0]


def initial_condition():
    u= np.zeros((NY,NX))
    u[NY//2,NX//2]= 1.0/(dx*dy)

    return u


def dudt(u):
    d2u_dx2= (np.roll(u,-1,axis=1) - 2*u+np.roll(u,1,axis=1))/dx**2
    d2u_dy2= (np.roll(u,-1,axis=0) - 2*u+np.roll(u,1,axis=0))/dy**2

    return DX*d2u_dx2 + DY*d2u_dy2


def rk4_step(u):

    k1= dudt(u)
    k2= dudt(u+0.5*DT*k1)
    k3= dudt(u+0.5*DT*k2)
    k4= dudt(u+DT*k3)

    return u + DT*(k1+2*k2+2*k3+k4)/6


def analytical_solution(t):
    return ( 1/(4*np.pi*t*np.sqrt(DX*DY)) * np.exp(-X**2/(4*DX*t)) * np.exp(-Y**2/(4*DY*t)))



u= initial_condition()
snapshots= {}
t= 0
for target_time in OUTPUT_TIMES:
    while t < target_time-1e-12:
        u= rk4_step(u)
        t+= DT
    snapshots[target_time]= u.copy()

for t,numerical in snapshots.items():

    analytical= analytical_solution(t)

    mass_numerical= np.sum(numerical)*dx*dy
    mass_analytical= np.sum(analytical)*dx*dy
    max_error= np.max(np.abs(numerical-analytical))
    l2_error= np.sqrt(np.mean((numerical-analytical)**2))
    relative_error= np.linalg.norm(numerical-analytical)/np.linalg.norm(analytical)*100

    print(f"t = {t}")
    print(f"Numerical Mass  = {mass_numerical:.10f}")
    print(f"Analytical Mass = {mass_analytical:.10f}")
    print(f"Max Error       = {max_error:.3e}")
    print(f"L2 Error        = {l2_error:.3e}")
    print(f"Relative Error  = {relative_error:.6f}%")
    print()

    zmax= max(np.max(numerical),np.max(analytical))

    fig= plt.figure(figsize=(12,5))
    ax1= fig.add_subplot(121,projection="3d")
    ax1.plot_surface(X,Y,numerical,cmap="Greys")
    ax1.set_title(f"Numerical (t={t})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    ax1.set_zlim(0,zmax)

    ax2= fig.add_subplot(122,projection="3d")
    ax2.plot_surface(X,Y,analytical,cmap="Greys")
    ax2.set_title(f"Analytical (t={t})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u")
    ax2.set_zlim(0,zmax)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"diffusion_2d_t{t:g}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

