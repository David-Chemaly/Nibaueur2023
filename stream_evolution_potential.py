import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

# Constants
G     = 4.3e-3 * u.pc * u.M_sun**-1 * (u.km/u.s)**2 # pc M_sun^-1 (km/s)^2
rho_c = 1.4e-7 * u.M_sun * u.pc**-3 # M_sun pc^-3

class NFW():

    def __init__(self,M,c,q):
        self.M = M
        self.c = c
        self.q = q

    def radius_flatten(self,x,y,z):
        return np.sqrt(x**2+y**2+(z/self.q)**2)
    
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
        A  = self.A_NFW()
        return - G/r * self.M/A * np.log(1 + r/Rs)
    
    # Outputs acceleration in km/s^2
    def acceleration(self,x,y,z):
        r  = self.radius_flatten(x,y,z)
        Rs = self.Rs_fct_RvirAndc()
        A  = self.A_NFW()
        a_r = G*self.M/A * (r/(r + Rs) - np.log(1 + r/Rs))/r**3

        a_x = a_r * x
        a_y = a_r * y
        a_z = a_r * z/self.q

        return [a_x.value,a_y.value,a_z.value] * a_x.unit
    
def acc_fct_RsAndc(x,y,z):
    M_vir = 4*np.pi/3*200*rho_c*(c*Rs)**3
    r = np.sqrt(x**2+y**2+(z/q)**2)

    a_r = G*M_vir/(np.log(1+c) - c/(1+c)) * (r/(r+Rs) - np.log(1+r/Rs))/r**3

    a_x = a_r * x
    a_y = a_r * y
    a_z = a_r * z/q

    return [a_x.value,a_y.value,a_z.value] * a_x.unit

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

        a_r = -G*self.M * (r**2 + self.a**2)**(-3/2)

        a_x = a_r * (x-xp)
        a_y = a_r * (y-yp)
        a_z = a_r * (z-zp)

        return [a_x.value,a_y.value,a_z.value] * a_x.unit
    
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

def LeepFrog_Coupled_all(a_fct, b_fct, xN_old, yN_old, zN_old,xp_old,yp_old,zp_old, vxp_old, vyp_old, vzp_old,vxN_old, vyN_old, vzN_old, dt):

    acc_p_old = a_fct(xp_old,yp_old,zp_old)
    acc_N_old = a_fct(xN_old,yN_old,zN_old) + b_fct(xN_old,yN_old,zN_old,xp_old,yp_old,zp_old) #a_fct(x_old,y_old,z_old)# b_fct(x_old,y_old,z_old,xp,yp,zp)

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
    acc_N_new = a_fct(xN_old,yN_old,zN_old) + b_fct(xN_new,yN_new,zN_new,xp_new,yp_new,zp_new) #a_fct(x_old,y_old,z_old)# b_fct(x_old,y_old,z_old,xp,yp,zp)

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
