import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 2D linear advection: u_t+cx u_x+cy u_y= 0
CX= 2
CY= 2
XMIN,XMAX= -20,20
YMIN,YMAX= -20,20
NX= 200
NY= 200
DT= 5e-3
SIGMA= 1.5
OUTPUT_TIMES= [0.1,0.3,1,3]

x= np.linspace(XMIN,XMAX,NX,endpoint=False)
y= np.linspace(YMIN,YMAX,NY,endpoint=False)

X,Y= np.meshgrid(x,y)
dx= x[1]-x[0]
dy= y[1]-y[0]


def initial_condition():
    u= np.exp(-(X**2+Y**2)/(2*SIGMA**2))
    return u/(np.sum(u)*dx*dy)


def dudt(u):
    du_dx= (np.roll(u,-1,axis=1) - np.roll(u,1,axis=1))/(2*dx)
    du_dy= (np.roll(u,-1,axis=0) - np.roll(u,1,axis=0))/(2*dy)

    return -CX*du_dx - CY*du_dy


def rk4_step(u): 

    k1= dudt(u)
    k2= dudt(u+0.5*DT*k1)
    k3= dudt(u+0.5*DT*k2)
    k4= dudt(u+DT*k3)

    return u + DT*(k1+2*k2+2*k3+k4)/6


def analytical_solution(t):

    u= np.exp(-((X-CX*t)**2+(Y-CY*t)**2)/(2*SIGMA**2))
    return u/(np.sum(u)*dx*dy)



u= initial_condition()
snapshots={}
t= 0

for target_time in OUTPUT_TIMES:
    while t < target_time-1e-12:
        u=rk4_step(u)
        t+=DT
    snapshots[target_time]=u.copy()

for t,numerical in snapshots.items():

    analytical=analytical_solution(t)

    mass_numerical= np.sum(numerical)*dx*dy
    mass_analytical= np.sum(analytical)*dx*dy

    print(f"t = {t}")
    print(f"Numerical Mass  = {mass_numerical:.10f}")
    print(f"Analytical Mass = {mass_analytical:.10f}")
    print()

    zmax= max(np.max(numerical),np.max(analytical))

    fig= plt.figure(figsize=(12,5))

    ax1= fig.add_subplot(121,projection="3d")
    ax1.plot_surface(X,Y,numerical,cmap="Greys")
    ax1.set_title(f"Numerical (t = {t})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    ax1.set_zlim(0,zmax)

    ax2= fig.add_subplot(122,projection="3d")
    ax2.plot_surface(X,Y,analytical,cmap="Greys")
    ax2.set_title(f"Analytical (t = {t})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u")
    ax2.set_zlim(0,zmax)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"advection_2d_central_t{t:g}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

