import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sympy as sp
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import minmax_scale

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML

# Functions

def phase(pos_N, pos_p):
    x, y, z = pos_N[:,0], pos_N[:,1], pos_N[:,2]
    xp, yp, zp = pos_p[0], pos_p[1], pos_p[2]
    return np.sqrt(x**2+y**2+z**2) - np.sqrt(xp**2+yp**2+zp**2)

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

class get_mat():

    def __init__(self, a, b, c, aa, bb):

        self.v1 = np.array([0, 0, 1])

        v2 = np.array([a, b, c])
        self.v2 = v2 / np.sum(v2**2)**.5

        v3 = np.cross(self.v1, self.v2)
        self.v3 = v3 / np.sum(v3**2)**.5

        self.aa = aa
        self.bb = bb
    
    def orientation(self):
        
        angle = np.arccos(np.sum(self.v1 * self.v2))

        return Rotation.from_rotvec(angle * self.v3).as_matrix()
    
    def rotation(self):

        z_new = - (self.aa * self.v2[0] + self.bb * self.v2[1]) / self.v2[2]
        v_new = np.array([self.aa, self.bb, z_new])

        v_norm = v_new / np.sum(v_new**2)**.5

        new_angle = np.arccos(np.sum(v_norm*self.v3)) 

        return Rotation.from_rotvec(new_angle * self.v2).as_matrix()
    
def run(mass_halo, r_s, q_xy, q_xz, alpha, beta, charlie, aa, bb, mass_plummer, r_plummer, time, dt, pos_p, vel_p, N, factor=1.1):

    # Rotate
    if alpha*beta*charlie*aa*bb != 0:
        rot_mat = get_mat(alpha, beta, charlie, aa, bb)
        R_orientation = rot_mat.orientation()
        R_rotation    = rot_mat.rotation()
    else:
        R_orientation = np.eye(3)
        R_rotation    = np.eye(3)

    # Define Main Halo Potential
    pot_NFW = gp.NFWPotential(mass_halo, r_s, a=1, b=q_xy, c=q_xz, units=galactic, origin=None, R=R_rotation@R_orientation)

    step   = int(time/dt)
    if N !=0:
        step_N = int(step/N)
    else:
        step_N = int(step/(N+1))

    orbit_pos_p = np.zeros((step, 3)) * u.kpc
    orbit_vel_p = np.zeros((step, 3)) 
    orbit_pos_p[0] = pos_p
    orbit_vel_p[0] = vel_p

    pos_N = np.zeros((N, 3)) * u.kpc
    vel_N = np.zeros((N, 3)) * u.km/u.s

    orbit_pos_N = np.zeros((step, N, 3)) * u.kpc
    orbit_vel_N = np.zeros((step, N, 3)) * u.km/u.s
    orbit_xhi_N = np.zeros((N, 1)) * u.kpc
    orbit_rt_N  = np.zeros((N, 1)) * u.kpc

    arg = []
    leading_arg  = []
    trailing_arg = []

    counter = 0
    for i in tqdm(range(step)):

        # Progenitor Phase Space Position
        wp = gd.PhaseSpacePosition(pos = pos_p,
                                   vel = vel_p)
        
        if i % step_N == 0 and N != 0:
            j = i//step_N
            no_rot_NFW = gp.NFWPotential(mass_halo, r_s, a=1, b=q_xy, c=q_xz, units=galactic, origin=None, R=None)
            rt     = get_rt(wp, no_rot_NFW, mass_plummer) * factor
            orbit_rt_N[i] = rt

            # if rt > 3*r_plummer:
            #     rt = 0

            rp     = np.linalg.norm( wp.xyz )
            theta  = np.arccos(wp.z/rp)
            phi    = np.arctan2(wp.y,wp.x)

            if counter%2 == 0:
                xt1, yt1, zt1 = (rp - rt)*np.sin(theta)*np.cos(phi), (rp - rt)*np.sin(theta)*np.sin(phi), (rp - rt)*np.cos(theta)
                leading_arg.append(i)
            else:
                xt1, yt1, zt1 = (rp + rt)*np.sin(theta)*np.cos(phi), (rp + rt)*np.sin(theta)*np.sin(phi), (rp + rt)*np.cos(theta)
                trailing_arg.append(i)
            arg.append(i)

            # New N starting position
            pos_N[j] = np.array([xt1.value, yt1.value, zt1.value]) * u.kpc #  # tidal radius

            # New N starting velocity
            sig = np.sqrt( G*mass_plummer/(6*np.sqrt(rt**2+r_plummer**2)) ).to(u.km/u.s)
            if counter%2 == 0:
                vel_N[j] = vel_p - np.sign(vel_p)*abs(np.random.normal(0, sig.value)) * u.km/u.s # velocity dispersion
            else:
                vel_N[j] = vel_p + np.sign(vel_p)*abs(np.random.normal(0, sig.value)) * u.km/u.s

            counter += 1

        # Save Progenitor new Position and Velocity
        orbit_pos_p[i] = pos_p
        orbit_vel_p[i] = vel_p

        if N != 0:

            # Save N new Position and Velocity
            orbit_pos_N[i] = pos_N
            orbit_vel_N[i] = vel_N

            orbit_xhi_N[:int(i+1), 0] += -phase(orbit_pos_N[i,:int(i+1)], pos_p)
            # orbit_rt_N[i] = rt
        
            # All N in Phase Space Position
            wN = gd.PhaseSpacePosition(pos = pos_N[:j+1].T,
                                    vel = vel_N[:j+1].T)

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
    

    gamma = minmax_scale(orbit_xhi_N[:,0].value, feature_range=(-1, 1))

    return orbit_pos_p, orbit_pos_N, gamma, arg, leading_arg, trailing_arg, orbit_rt_N

if __name__ == '__main__':

    # Parameters
    mass_halo = 1e12 * u.Msun
    r_s = 15 * u.kpc
    q_xy = 1
    q_xz = 1

    alpha, beta, charlie = 1, 1, 0
    aa, bb = 1, 1

    mass_plummer = 1e8 * u.Msun
    r_plummer = 1 * u.kpc

    time = 4 * u.Gyr
    dt   = 1 * u.Myr

    pos_p = [-70, 0, 0] * u.kpc
    vel_p = [0, 175, 0] * u.km/u.s

    N  = 4000

    orbit_pos_p, orbit_pos_N, gamma, arg, leading_arg, trailing_arg, orbit_rt_N = run(mass_halo, r_s, q_xy, q_xz, 
                                                                                      alpha, beta, charlie, aa, bb, 
                                                                                      mass_plummer, r_plummer, 
                                                                                      time, dt, 
                                                                                      pos_p, vel_p, 
                                                                                      N)

    axis_1 = 0
    axis_2 = 1
    zoom   = False

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Set up formatting for the movie files
    num_frames = 100
    def generate_data(frame_number):
        idx = np.linspace(0, int(time/dt)-1, num_frames, dtype=int)[frame_number]
        return orbit_pos_N[idx, :, :], orbit_rt_N[idx, 0]

    def update(frame_number):

        # Frame index
        idx = np.linspace(0, int(time/dt)-1, num_frames, dtype=int)[frame_number]

        # Orbit
        x = orbit_pos_p[:,axis_1]
        y = orbit_pos_p[:,axis_2]

        # Stream
        data, here_rt = generate_data(frame_number)
        xN = data[:,axis_1]
        yN = data[:,axis_2]

        # Set limits
        if zoom == False:
            x_limits = (-115,115)
            y_limits = (-90,90)
        elif zoom == True:
            deltav = 15
            x_limits = (x[idx]-deltav,x[idx]+deltav) 
            y_limits = (y[idx]-deltav,y[idx]+deltav) 


        ax.clear()
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect('equal')
        ax.set_title(f'{idx * dt.to(u.Gyr).value:.2f} Gyr')
        ax.set_xlabel('X (kpc)')
        ax.set_ylabel('Y (kpc)')

        im1 = ax.plot(x, y, label='Orbit')
        im2 = ax.scatter(xN[arg], yN[arg], color='b', s=1, label='Stream')
        # im2 = ax.scatter(xN[leading_arg], yN[leading_arg], color='orange', s=1, label='Leading')
        # im3 = ax.scatter(xN[trailing_arg], yN[trailing_arg], color='b', s=1, label='Trailing')

        im3 = ax.scatter(0, 0, color='k', label='Center')
        im4 = ax.scatter(x[0], y[0], color='g', label='Start')
        im5 = ax.scatter(x[-1], y[-1], color='r', label='End')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        im6 = ax.text(0.05, 0.95, f'r$_t$ = {here_rt:.02f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=props)

        ax.legend(loc='upper right')

        return im1 + [im2, im3, im4, im5, im6]


    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_frames, blit=True)
    writervideo = FFMpegWriter(fps=10) 
    anim.save('./animation_lowsig.mp4', writer=writervideo)
    plt.show()

