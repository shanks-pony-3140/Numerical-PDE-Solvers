import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 2D linear advection: u_t+cx u_x+cy u_y= 0
CX= 2
CY= 1
XMIN,XMAX= -20,20
YMIN,YMAX= -20,20
NX= 200
NY= 200
CFL= 0.5
SIGMA= 1.0
OUTPUT_TIMES= [0.1,0.3,1,3]

x= np.linspace(XMIN,XMAX,NX,endpoint=False)
y= np.linspace(YMIN,YMAX,NY,endpoint=False)
X,Y= np.meshgrid(x,y)
dx= x[1]-x[0]
dy= y[1]-y[0]
dt= CFL/(abs(CX)/dx+abs(CY)/dy)


def initial_condition():
    u= np.exp(-(X**2+Y**2)/(2*SIGMA**2))
    return u/(np.sum(u)*dx*dy)


def dudt(u):
    # Rusanov numerical fluxes in x and y with periodic boundaries.
    u_right_x= np.roll(u,-1,axis=1)
    flux_x= 0.5*CX*(u+u_right_x) - 0.5*abs(CX)*(u_right_x-u)

    u_right_y= np.roll(u,-1,axis=0)
    flux_y= 0.5*CY*(u+u_right_y) - 0.5*abs(CY)*(u_right_y-u)

    div_flux_x= (flux_x-np.roll(flux_x,1,axis=1))/dx
    div_flux_y= (flux_y-np.roll(flux_y,1,axis=0))/dy
    return -(div_flux_x + div_flux_y)


def tvd_rk3_step(u):

    k1= dudt(u)
    u1= u+dt*k1

    k2= dudt(u1)
    u2= 0.75*u + 0.25*(u1+dt*k2)

    k3= dudt(u2)
    return (1/3)*u + (2/3)*(u2+dt*k3)


def analytical_solution(t):
    u= np.exp(-((X-CX*t)**2 + (Y-CY*t)**2)/(2*SIGMA**2))
    return u/(np.sum(u)*dx*dy)



u= initial_condition()
snapshots= {}
t= 0.0

for target_time in OUTPUT_TIMES:
    while t < target_time-1e-12:
        u= tvd_rk3_step(u)
        t+= dt
    snapshots[target_time]= u.copy()

for t,numerical in snapshots.items():
    analytical= analytical_solution(t)
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
    plt.savefig(FIGURES_DIR / f"advection_2d_rusanov_t{t:g}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


