import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
r_s = 20 * u.kpc

mass_plummer = 1e10 * u.Msun
r_plummer = 2 * u.kpc

time = 1 * u.Gyr
dt   = 1 * u.Myr

pos_p = [-50, 0, 1] * u.kpc
vel_p = [0, 175, 0] * u.km/u.s

N  = 999 # has to be smaller than step
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

for i in tqdm(range(step-1)):

    if i % step_N == 0:
        j = i//step_N

        # New N initial conditions
        pos_N[j] = pos_p + 5 * u.kpc # tidal radius
        vel_N[j] = vel_p # + 100 * u.km/u.s # velocity dispersion

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

    # Progenitor Phase Space Position
    wp = gd.PhaseSpacePosition(pos = pos_p,
                               vel = vel_p)
    
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
plt.subplot(1,2,1)
plt.scatter(orbit_pos_p[:, 0], orbit_pos_p[:, 1], s=1, c='k')
plt.scatter(orbit_pos_N[-1, :, 0], orbit_pos_N[-1, :, 1], s=1, c='r')
plt.subplot(1,2,2)
plt.scatter(orbit_pos_p[:, 0], orbit_pos_p[:, 1], s=1, c='k')
for i in range(N):
    plt.scatter(orbit_pos_N[:, i, 0], orbit_pos_N[:, i, 1], s=1)
plt.show()