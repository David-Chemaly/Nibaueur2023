import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sympy as sp

import astropy.units as u
from astropy.constants import G
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

import gala.potential as gp
import gala.dynamics as gd
import gala.integrate as gi
import gala.units as gu

from gala.units import galactic
from gala.potential import NFWPotential
from gala.dynamics import PhaseSpacePosition, MockStream
from gala.integrate import LeapfrogIntegrator

# Parameters

mass_halo = 1e12 * u.Msun
r_s = 10 * u.kpc

mass_plummer = 1e8 * u.Msun
r_plummer = 1 * u.kpc

time = 4 * u.Gyr
dt   = 1 * u.Myr

pos_p = [-50, 0, 0] * u.kpc
vel_p = [0, 175, 0] * u.km/u.s

N  = 3999 # has to be smaller than step
dN = 1

# Define Main Halo Potential
pot_NFW = gp.NFWPotential(mass_halo, r_s, a=1, b=1, c=1, units=galactic, origin=None, R=None)

step = int(time/dt)
step_N = int(step/N)

orbit_pos_p = np.zeros((step, 3)) * u.kpc
orbit_vel_p = np.zeros((step, 3)) 
orbit_pos_p[0] = pos_p
orbit_vel_p[0] = vel_p

pos_N = np.zeros((N, 3)) * u.kpc
vel_N = np.zeros((N, 3)) * u.km/u.s

orbit_pos_N = np.zeros((step, N, 3)) * u.kpc
orbit_vel_N = np.zeros((step, N, 3)) * u.km/u.s

# Functions

def get_Jacobian(a,b,c):
    # Define the symbols for Cartesian and Spherical coordinates
    x, y, z = sp.symbols('x y z')

    # Define the transformations from Cartesian to Spherical coordinates
    r_expr = sp.sqrt(x**2 + y**2 + z**2)
    theta_expr = sp.acos(z / sp.sqrt(x**2 + y**2 + z**2))
    phi_expr = sp.atan2(y, x)

    # Create the Jacobian matrix
    J = sp.Matrix([
        [r_expr.diff(x), r_expr.diff(y), r_expr.diff(z)],
        [theta_expr.diff(x), theta_expr.diff(y), theta_expr.diff(z)],
        [phi_expr.diff(x), phi_expr.diff(y), phi_expr.diff(z)]
    ])

    # Define a specific point (x, y, z)
    point = {x: a, y: b, z: c}  # Example point

    # Substitute the point into the Jacobian matrix to get numeric values
    J_numeric = J.subs(point)

    return np.array(J_numeric, dtype=float)

def get_rt(wp, pot_NFW, mass_plummer):

    rp = np.linalg.norm( wp.xyz )
    angular_velocity = ( np.linalg.norm( wp.angular_momentum() ) / rp**2 ).to(u.Gyr**-1)

    J = get_Jacobian(wp.x.value, wp.y.value, wp.z.value)
    d2pdr2 = (J.T * pot_NFW.hessian( wp )[:,:,0] * J)[0,0]
    rt = ( G * mass_plummer / (angular_velocity**2 - d2pdr2) ).to(u.kpc**3) **(1/3)
    return rt

leading_arg  = []
trailing_arg = []
for i in tqdm(range(step-1)):

    # Progenitor Phase Space Position
    wp = gd.PhaseSpacePosition(pos = pos_p,
                               vel = vel_p)
    
    if i % step_N == 0:
        j = i//step_N

        rt     = get_rt(wp, pot_NFW, mass_plummer) * .5
        rp     = np.linalg.norm( wp.xyz )
        theta  = np.arccos(wp.z/rp)
        phi    = np.arctan2(wp.y,wp.x)

        if i%2 == 0:
            xt1, yt1, zt1 = (rp - rt)*np.sin(theta)*np.cos(phi), (rp - rt)*np.sin(theta)*np.sin(phi), (rp - rt)*np.cos(theta)
            leading_arg.append(i)
        else:
            xt1, yt1, zt1 = (rp + rt)*np.sin(theta)*np.cos(phi), (rp + rt)*np.sin(theta)*np.sin(phi), (rp + rt)*np.cos(theta)
            trailing_arg.append(i)

        sig = np.sqrt(G*mass_plummer/(6*np.sqrt(rt**2+r_plummer**2))).to(u.km/u.s)

        # New N initial conditions
        pos_N[j] = np.array([xt1.value, yt1.value, zt1.value]) * u.kpc #  # tidal radius
        # direction = np.array([xt1.value - wp.x.value, yt1.value - wp.y.value, zt1.value - wp.z.value])
        # direction = -direction / np.linalg.norm(direction)
        vel_N[j] = vel_p #+ direction*np.random.normal(0, sig.value) * u.km/u.s # velocity dispersion

    if i == 0:
        orbit_pos_N[i] = pos_N
        orbit_vel_N[i] = vel_N

    # All N in Phase Space Position
    wN = gd.PhaseSpacePosition(pos = pos_N[:j + 1].T,
                               vel = vel_N[:j + 1].T)

    # Define Plummer Potential
    pot_plummer  = gp.PlummerPotential(mass_plummer, r_plummer, units=galactic, origin=pos_p, R=None)
    pot_combined = pot_NFW + pot_plummer
    orbit_N = gp.Hamiltonian(pot_combined).integrate_orbit(wN, dt=dt, n_steps=1)
    pos_N[:j+1] = orbit_N.xyz[:, -1].T
    vel_N[:j+1] = orbit_N.v_xyz[:, -1].T
    
    # Progenitor new Position and Velocity
    orbit_p = gp.Hamiltonian(pot_NFW).integrate_orbit(wp, dt=dt, n_steps=1)
    pos_p = orbit_p.xyz[:, -1]
    vel_p = orbit_p.v_xyz[:, -1]

    # Save Progenitor new Position and Velocity
    orbit_pos_p[i+1] = pos_p
    orbit_vel_p[i+1] = vel_p

    # Save N new Position and Velocity
    orbit_pos_N[i+1] = pos_N
    orbit_vel_N[i+1] = vel_N

plt.figure(figsize=(10,5))
plot_step = 4
plot_time = np.linspace(0, time.value, plot_step)
for i in range(len(plot_time)):
    plt.subplot(2,2,i+1)
    idx = step//4 * (i+1) - 1
    plt.title(f'{time.value/plot_step * (i+1):.2f} Gyr')
    plt.scatter(orbit_pos_p[:idx, 0], orbit_pos_p[:idx, 1], s=1, c='k')
    plt.scatter(orbit_pos_N[idx, leading_arg, 0], orbit_pos_N[idx, leading_arg, 1], s=1, c='r', label = 'Leading')
    plt.scatter(orbit_pos_N[idx, trailing_arg, 0], orbit_pos_N[idx, trailing_arg, 1], s=1, c='b', label = 'Trailing')
    plt.scatter(0,0,color='orange', label = 'Center')
    plt.scatter(orbit_pos_p[0,0],orbit_pos_p[0,1],color='g', label = 'Start')
    plt.scatter(orbit_pos_p[idx,0],orbit_pos_p[idx,1],color='k', label = 'End')
    if i == 0:
        plt.legend()
    if i == 0 or i == 2:
        plt.ylabel('kpc')
    if i == 2 or i == 3:
        plt.xlabel('kpc')
    plt.axis('equal')
plt.show()

# plt.subplot(1,2,2)
# plt.scatter(orbit_pos_p[:, 0], orbit_pos_p[:, 1], s=1, c='k')
# for i in range(N):
#     plt.scatter(orbit_pos_N[:, i, 0], orbit_pos_N[:, i, 1], s=1)