import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from tqdm import tqdm
import torch

from astropy.constants import G
from astropy.cosmology import default_cosmology
cosmo = default_cosmology.get()
rho_c = (3 * cosmo.H(0.0) ** 2 / (8 * np.pi * G)).to(u.Msun / u.kpc ** 3)

class NFW():

    def __init__(self,M,c,qx,qy,qz):
        self.M = M
        self.c = c
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def radius_flatten(self,x,y,z):
        return np.sqrt((x/self.qx)**2+(y/self.qy)**2+(z/self.qz)**2)
    
    def A_NFW(self):
        return np.log(1+self.c) - self.c/(1+self.c)
    
    def rho0_Rscube(self):
        return self.M/self.A_NFW()
    
    # Convention 200
    def Rvir_fct_M(self):
        return (3*self.M/(4*np.pi*200*rho_c))**(1/3) 
    
    def Rs_fct_RvirAndc(self):
        return self.Rvir_fct_M()/self.c
        
    # Outputs potential in (km/s)^2
    def potential(self,x,y,z):
        r  = self.radius_flatten(x,y,z)

        Rs = self.Rs_fct_RvirAndc()

        return - G/r * self.M/self.A_NFW() * np.log(1 + r/Rs)
    
    # Outputs acceleration in km/s^2
    def acceleration(self,x,y,z):
        r   = self.radius_flatten(x,y,z)
        Rs  = self.Rs_fct_RvirAndc()

        a_r = -G*self.M/(self.A_NFW()*r**2*(r+Rs)) * ( (r+Rs)*np.log(1 + r/Rs) - r)

        a_x = a_r/r * x/self.qx
        a_y = a_r/r * y/self.qy
        a_z = a_r/r * z/self.qz

        return [a_x.value,a_y.value,a_z.value] * a_x.unit
    
    # Outputs second derivative of the potential in 1/s^2
    def second_derivative_potential(self,x,y,z):
        r   = self.radius_flatten(x,y,z)
        Rs  = self.Rs_fct_RvirAndc()

        return G*self.M/(self.A_NFW()*r**3*(r+Rs)**2) * ( (2*r**2+4*Rs*r+2*Rs**2) * np.log(1 + r/Rs) - 3*r**2 - 2*Rs*r)

    def mass_enclosed(self,r):
        return self.M * (np.log(1 + r/self.Rs_fct_RvirAndc()) - r/(r + self.Rs_fct_RvirAndc()))


class Plummer():

    def __init__(self,M,a):
        self.M = M
        self.a = a

    def radius(self,x,y,z,xp,yp,zp):
        return np.sqrt((x-xp)**2+(y-yp)**2+(z-zp)**2)
    
    def potential(self,x,y,z,xp,yp,zp):
        r = self.radius(x,y,z,xp,yp,zp)
        return -G*self.M/np.sqrt(r**2+self.a**2)
    
    def acceleration(self,x,y,z,xp,yp,zp):
        r = self.radius(x,y,z,xp,yp,zp)

        a_r = -G*self.M*r * (r**2 + self.a**2)**(-3/2)

        a_x = a_r/r * (x-xp)
        a_y = a_r/r * (y-yp)
        a_z = a_r/r * (z-zp)

        return [a_x.value,a_y.value,a_z.value] * a_x.unit
    
    def mass_enclosed(self,r):
        return self.M
    
def LeepFrog(a_fct, x_old, y_old, z_old, vx_old, vy_old, vz_old, dt):

    acc_old = a_fct(x_old,y_old,z_old)

    acc_x_old = acc_old[0].to(u.km/u.s**2)
    acc_y_old = acc_old[1].to(u.km/u.s**2)
    acc_z_old = acc_old[2].to(u.km/u.s**2)
    
    vx_half = vx_old + 0.5*dt*acc_x_old
    vy_half = vy_old + 0.5*dt*acc_y_old
    vz_half = vz_old + 0.5*dt*acc_z_old

    x_new = x_old + dt*vx_half
    y_new = y_old + dt*vy_half
    z_new = z_old + dt*vz_half

    acc_new = a_fct(x_new,y_new,z_new)

    acc_x_new = acc_new[0].to(u.km/u.s**2)
    acc_y_new = acc_new[1].to(u.km/u.s**2)
    acc_z_new = acc_new[2].to(u.km/u.s**2)

    vx_new = vx_half + 0.5*dt*acc_x_new
    vy_new = vy_half + 0.5*dt*acc_y_new
    vz_new = vz_half + 0.5*dt*acc_z_new

    return x_new, y_new, z_new, vx_new, vy_new, vz_new

def LeepFrog_Coupled_halo(a_fct, xN_old, yN_old, zN_old,xp_old,yp_old,zp_old, vxp_old, vyp_old, vzp_old,vxN_old, vyN_old, vzN_old, dt):

    acc_p_old = a_fct(xp_old,yp_old,zp_old)
    acc_N_old = a_fct(xN_old,yN_old,zN_old)

    acc_x_old_p = acc_p_old[0].to(u.km/u.s**2)
    acc_y_old_p = acc_p_old[1].to(u.km/u.s**2)
    acc_z_old_p = acc_p_old[2].to(u.km/u.s**2)

    acc_x_old_N = acc_N_old[0].to(u.km/u.s**2)
    acc_y_old_N = acc_N_old[1].to(u.km/u.s**2)
    acc_z_old_N = acc_N_old[2].to(u.km/u.s**2)
    
    vxp_half = vxp_old + 0.5*dt*acc_x_old_p
    vyp_half = vyp_old + 0.5*dt*acc_y_old_p
    vzp_half = vzp_old + 0.5*dt*acc_z_old_p

    vxN_half = vxN_old + 0.5*dt*acc_x_old_N
    vyN_half = vyN_old + 0.5*dt*acc_y_old_N
    vzN_half = vzN_old + 0.5*dt*acc_z_old_N
    
    xp_new = xp_old + dt*vxp_half
    yp_new = yp_old + dt*vyp_half
    zp_new = zp_old + dt*vzp_half

    xN_new = xN_old + dt*vxN_half
    yN_new = yN_old + dt*vyN_half
    zN_new = zN_old + dt*vzN_half

    acc_p_new = a_fct(xp_new,yp_new,zp_new)
    acc_N_new = a_fct(xN_old,yN_old,zN_old)

    acc_x_new_p = acc_p_new[0].to(u.km/u.s**2)
    acc_y_new_p = acc_p_new[1].to(u.km/u.s**2)
    acc_z_new_p = acc_p_new[2].to(u.km/u.s**2)

    acc_x_new_N = acc_N_new[0].to(u.km/u.s**2)
    acc_y_new_N = acc_N_new[1].to(u.km/u.s**2)
    acc_z_new_N = acc_N_new[2].to(u.km/u.s**2)


    vxp_new = vxp_half + 0.5*dt*acc_x_new_p
    vyp_new = vyp_half + 0.5*dt*acc_y_new_p
    vzp_new = vzp_half + 0.5*dt*acc_z_new_p

    vxN_new = vxN_half + 0.5*dt*acc_x_new_N
    vyN_new = vyN_half + 0.5*dt*acc_y_new_N
    vzN_new = vzN_half + 0.5*dt*acc_z_new_N

    return xp_new,yp_new,zp_new,vxp_new,vyp_new,vzp_new,xN_new,yN_new,zN_new,vxN_new,vyN_new,vzN_new

# halo.acceleration, 
# progenitor.acceleration,
# all_pos_N[0,:int((t+1)*N)], all_pos_N[1,:int((t+1)*N)], all_pos_N[2,:int((t+1)*N)],
# xp,yp,zp, 
# vxp,vyp, vzp,
# all_vel_N[0,:int((t+1)*N)], all_vel_N[1,:int((t+1)*N)], all_vel_N[2,:int((t+1)*N)], 
# dt * u.Gyr)

def LeepFrog_Coupled_all(a_fct, b_fct, xN_old, yN_old, zN_old,xp_old,yp_old,zp_old, vxp_old, vyp_old, vzp_old,vxN_old, vyN_old, vzN_old, dt):

    acc_p_old = a_fct(xp_old,yp_old,zp_old)
    acc_N_old = a_fct(xN_old,yN_old,zN_old) + b_fct(xN_old,yN_old,zN_old,xp_old,yp_old,zp_old) 

    acc_x_old_p = acc_p_old[0].to(u.km/u.s**2)
    acc_y_old_p = acc_p_old[1].to(u.km/u.s**2)
    acc_z_old_p = acc_p_old[2].to(u.km/u.s**2)

    acc_x_old_N = acc_N_old[0].to(u.km/u.s**2)
    acc_y_old_N = acc_N_old[1].to(u.km/u.s**2)
    acc_z_old_N = acc_N_old[2].to(u.km/u.s**2)
    
    vxp_half = vxp_old + 0.5*dt*acc_x_old_p
    vyp_half = vyp_old + 0.5*dt*acc_y_old_p
    vzp_half = vzp_old + 0.5*dt*acc_z_old_p

    vxN_half = vxN_old + 0.5*dt*acc_x_old_N
    vyN_half = vyN_old + 0.5*dt*acc_y_old_N
    vzN_half = vzN_old + 0.5*dt*acc_z_old_N
    
    xp_new = xp_old + dt*vxp_half
    yp_new = yp_old + dt*vyp_half
    zp_new = zp_old + dt*vzp_half

    xN_new = xN_old + dt*vxN_half
    yN_new = yN_old + dt*vyN_half
    zN_new = zN_old + dt*vzN_half

    acc_p_new = a_fct(xp_new,yp_new,zp_new)
    acc_N_new = a_fct(xN_old,yN_old,zN_old) + b_fct(xN_new,yN_new,zN_new,xp_new,yp_new,zp_new) 

    acc_x_new_p = acc_p_new[0].to(u.km/u.s**2)
    acc_y_new_p = acc_p_new[1].to(u.km/u.s**2)
    acc_z_new_p = acc_p_new[2].to(u.km/u.s**2)

    acc_x_new_N = acc_N_new[0].to(u.km/u.s**2)
    acc_y_new_N = acc_N_new[1].to(u.km/u.s**2)
    acc_z_new_N = acc_N_new[2].to(u.km/u.s**2)


    vxp_new = vxp_half + 0.5*dt*acc_x_new_p
    vyp_new = vyp_half + 0.5*dt*acc_y_new_p
    vzp_new = vzp_half + 0.5*dt*acc_z_new_p

    vxN_new = vxN_half + 0.5*dt*acc_x_new_N
    vyN_new = vyN_half + 0.5*dt*acc_y_new_N
    vzN_new = vzN_half + 0.5*dt*acc_z_new_N

    return xp_new,yp_new,zp_new,vxp_new,vyp_new,vzp_new,xN_new,yN_new,zN_new,vxN_new,vyN_new,vzN_new


def phase(x,y,z,xp,yp,zp):
    return np.sqrt(x**2+y**2+z**2) - np.sqrt(xp**2+yp**2+zp**2)

def L1L2(halo, progenitor, xp, yp, zp, N=1000, delta=10):
    
    r_prog = np.sqrt(xp**2+yp**2+zp**2).value
    theta  = np.arccos(zp.value/r_prog)
    phi    = np.arctan2(yp.value,xp.value)

    all_r  = np.linspace(r_prog - delta, r_prog + delta, N)

    all_x = all_r * np.sin(theta) * np.cos(phi)
    all_y = all_r * np.sin(theta) * np.sin(phi)
    all_z = all_r * np.cos(theta)

    all_acc_halo = halo.acceleration(all_x *u.kpc, all_y *u.kpc, all_z *u.kpc).value
    all_acc_prog = progenitor.acceleration(all_x *u.kpc, all_y *u.kpc, all_z *u.kpc, xp, yp, zp).value

    idx_L = np.argmin(abs(np.linalg.norm(all_acc_halo, axis=0) - np.linalg.norm(all_acc_prog, axis=0)))
    dif_L = N//2 - idx_L

    L1 = np.array([all_x[N//2 - dif_L], all_y[N//2 - dif_L], all_z[N//2 - dif_L]])[:,None]
    L2 = np.array([all_x[N//2 + dif_L], all_y[N//2 + dif_L], all_z[N//2 + dif_L]])[:,None]

    return L1, L2

def r_t(rp,halo,progenitor):
    return rp * (progenitor.mass_enclosed(rp) / (3*halo.mass_enclosed(rp)) )**(1/3)

def run(t_start, t_end, dt, halo, progenitor, pos_prog, vel_prog, vel_scat, N):

    time    = np.arange(t_start + dt, t_end + dt, dt)

    xp, yp, zp    = pos_prog[0] * u.kpc, pos_prog[1] * u.kpc, pos_prog[2] * u.kpc # kpc

    vxp, vyp, vzp = vel_prog[0] * u.km/u.s, vel_prog[1] * u.km/u.s, vel_prog[2] * u.km/u.s # km/s
    vx_scatter, vy_scatter, vz_scatter = vel_scat[0] * u.km/u.s, vel_scat[1] * u.km/u.s, vel_scat[2] * u.km/u.s # km/s

    all_pos_p = np.zeros([3, len(time)+2]) * u.kpc
    all_vel_p = np.zeros([3, len(time)+2]) * u.km/ u.s
    all_pos_p[:,0] = [xp,yp,zp] 
    all_vel_p[:,0] = [vxp,vyp,vzp]

    all_pos_N = np.zeros([3, N*len(time)+2]) * u.kpc
    all_vel_N = np.zeros([3, N*len(time)+2]) * u.km/u.s
    all_xhi_N = np.zeros([1, N*len(time)+2]) * u.kpc
    
    save_all_pos_N = np.zeros([len(time)+1, 3, N*len(time)+2]) * u.kpc
    save_all_vel_N = np.zeros([len(time)+1, 3, N*len(time)+2]) * u.km/u.s
    save_all_xhi_N = np.zeros([len(time)+1, 1, N*len(time)+2]) * u.kpc

    all_rt = np.zeros(len(time)+1) * u.kpc
    all_L1 = np.zeros([3,len(time)+1]) * u.kpc
    all_L2 = np.zeros([3,len(time)+1]) * u.kpc

    for t in tqdm(range(len(time)+1), leave=True):

        r_prog = np.sqrt(xp**2+yp**2+zp**2)
        rt = r_t(r_prog, halo, progenitor)

        all_rt[t] = rt #* u.kpc
        theta  = np.arccos(zp/r_prog)
        phi    = np.arctan2(yp,xp)
        xt1, yt1, zt1 = (r_prog - rt)*np.sin(theta)*np.cos(phi), (r_prog - rt)*np.sin(theta)*np.sin(phi), (r_prog - rt)*np.cos(theta)
        xt2, yt2, zt2 = (r_prog + rt)*np.sin(theta)*np.cos(phi), (r_prog + rt)*np.sin(theta)*np.sin(phi), (r_prog + rt)*np.cos(theta)

        all_L1[:,t] = [xt1, yt1, zt1]
        all_L2[:,t] = [xt2, yt2, zt2]

        xN_L1, yN_L1, zN_L1    = xt1.value*torch.ones(N//2), yt1.value*torch.ones(N//2), zt1.value*torch.ones(N//2)
        xN_L2, yN_L2, zN_L2    = xt2.value*torch.ones(N - N//2), yt2.value*torch.ones(N - N//2), zt2.value*torch.ones(N - N//2)

        xN, yN, zN    = torch.cat((xN_L1,xN_L2))*u.kpc, torch.cat((yN_L1,yN_L2))*u.kpc, torch.cat((zN_L1,zN_L2))*u.kpc
        vxN, vyN, vzN =  vxp + torch.randn(N)*vx_scatter, vyp + torch.randn(N)*vy_scatter, vzp + torch.randn(N)*vz_scatter

        # x_scatter, y_scatter, z_scatter = [1, 1, 1] * u.kpc
        # xN, yN, zN    = xp + torch.randn(N)*x_scatter, yp + torch.randn(N)*y_scatter, zp + torch.randn(N)*z_scatter
        # vxN, vyN, vzN = vxp + torch.randn(N)*vx_scatter, vyp + torch.randn(N)*vy_scatter, vzp + torch.randn(N)*vz_scatter

        all_pos_N[:,int(t*N):int((t+1)*N)] = [xN,yN,zN] 
        all_vel_N[:,int(t*N):int((t+1)*N)] = [vxN,vyN,vzN]

        if progenitor == None:
            xp,yp,zp,vxp,vyp,vzp,xN,yN,zN,vxN,vyN,vzN = LeepFrog_Coupled_halo(halo.acceleration, 
                                                                            all_pos_N[0,:int((t+1)*N)], all_pos_N[1,:int((t+1)*N)], all_pos_N[2,:int((t+1)*N)],
                                                                            xp,yp,zp, 
                                                                            vxp,vyp, vzp,
                                                                            all_vel_N[0,:int((t+1)*N)], all_vel_N[1,:int((t+1)*N)], all_vel_N[2,:int((t+1)*N)], 
                                                                            dt * u.Gyr)
        
        elif progenitor != None:
            xp,yp,zp,vxp,vyp,vzp,xN,yN,zN,vxN,vyN,vzN = LeepFrog_Coupled_all(halo.acceleration, 
                                                                            progenitor.acceleration,
                                                                            all_pos_N[0,:int((t+1)*N)], all_pos_N[1,:int((t+1)*N)], all_pos_N[2,:int((t+1)*N)],
                                                                            xp,yp,zp, 
                                                                            vxp,vyp, vzp,
                                                                            all_vel_N[0,:int((t+1)*N)], all_vel_N[1,:int((t+1)*N)], all_vel_N[2,:int((t+1)*N)], 
                                                                            dt * u.Gyr)
            
        # Update 
        all_pos_N[:,:int((t+1)*N)] = [xN,yN,zN] 
        all_vel_N[:,:int((t+1)*N)] = [vxN,vyN,vzN]
        all_xhi_N[:,:int((t+1)*N)] += phase(all_pos_N[0,:int((t+1)*N)],
                                        all_pos_N[1,:int((t+1)*N)],
                                        all_pos_N[2,:int((t+1)*N)],
                                        xp,yp,zp)
        
        all_pos_p[:,(t+1)] = [xp,yp,zp] 
        all_vel_p[:,(t+1)] = [vxp,vyp,vzp]

        # Save
        save_all_pos_N[t,:,:int((t+1)*N)] = all_pos_N[:,:int((t+1)*N)].copy()
        save_all_vel_N[t,:,:int((t+1)*N)] = all_vel_N[:,:int((t+1)*N)].copy()
        save_all_xhi_N[t,:,:int((t+1)*N)] = all_xhi_N[:,:int((t+1)*N)].copy()



    return time, all_pos_p.value, all_vel_p.value, save_all_pos_N.value, save_all_vel_N.value, save_all_xhi_N.value, all_rt.value, all_L1.value, all_L2.value
