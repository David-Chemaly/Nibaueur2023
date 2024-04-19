import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic

import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')

# Get stream from Gala
def get_stream_Fardal(mass_halo, r_s, q_xy, q_xz, theta, phi, zz,
                      mass_plummer, r_plummer, 
                      pos_p, vel_p, 
                      time):
    
    pot = gp.NFWPotential(mass_halo, 
                        r_s, 
                        a=1, b=q_xy, c=q_xz, 
                        units=galactic, 
                        origin=None, 
                        R=None)

    H = gp.Hamiltonian(pot)

    prog_w0 = gd.PhaseSpacePosition(pos=pos_p,
                                    vel=vel_p)

    df = ms.FardalStreamDF(gala_modified=True)

    prog_pot = gp.PlummerPotential(m=mass_plummer, 
                                b=r_plummer, 
                                units=galactic)

    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)

    dt = 1 * u.Myr
    stream, prog = gen.run(prog_w0, 
                        mass_plummer,
                        dt=dt, 
                        n_steps=time/dt)
    
    return stream, prog

if __name__ == '__main__':

    # Define 16 parameters
    mass_halo = 1e12 * u.Msun
    r_s       = 15 * u.kpc
    q_xy      = 1
    q_xz      = 1
    theta, phi, zz = 0, 0, 0

    mass_plummer = 1e8 * u.Msun
    r_plummer    = 1 * u.kpc

    pos_p = [-70, 0, 0] * u.kpc
    vel_p = [0, 175, 0] * u.km/u.s

    time = 4 * u.Gyr

    # Get stream from Gala 
    stream, prog = get_stream_Fardal(mass_halo, r_s, q_xy, q_xz, theta, phi, zz,
                                     mass_plummer, r_plummer, 
                                     pos_p, vel_p, 
                                     time)

    plt.scatter(stream.xyz[0], stream.xyz[1], s=.1, c='b')
    plt.show()
