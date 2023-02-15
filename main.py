# Packages
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import InterpolatedUnivariateSpline
import sympy as sym

#plt.close('all')

#--- Functions ---#

# Runge-Kutta of order 4
def rk4(f, initial, t0, tf, n):
    t = np.linspace(t0, tf, n)
    vars = np.zeros(shape=(4, n))
    vars[0, 0] = initial[0] #x
    vars[1, 0] = initial[1] #z
    vars[2, 0] = initial[2] #kx
    vars[3, 0] = initial[3] #kz
    h = t[1]-t[0]
    for i in range(1, n):
        k1 = f(*vars[:, i-1])
        k2 = f(*vars[:, i-1] + 0.5 * k1*h)
        k3 = f(*vars[:, i-1] + 0.5 * k2*h)
        k4 = f(*vars[:, i-1] + k3*h)
        vars[:, i] = vars[:, i-1] + (k1 + 2*k2 + 2*k3 + k4)*h/6
    return vars, t

# Problem function
def problem(xr,zr,kxr,kzr):
    # To avoid interpolation we look for the index closest to z of the Runge-Kutta
    index=np.argmin(np.abs(z-zr))
    dkx_ds = 0
    dkz_ds = 2.0*N[index]*Nz[index]*cs[index]**2*kxr**2/np.sqrt(-4*N[index]**2*cs[index]**2*kxr**2 + (cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2)**2) - csz[index]*(cs[index]*(kxr**2 + kzr**2) + 0.5*(-4*N[index]**2*cs[index]*kxr**2 + 2*cs[index]*(kxr**2 + kzr**2)*(cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2))/np.sqrt(-4*N[index]**2*cs[index]**2*kxr**2 + (cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2)**2)) - wcz[index]*(1.0*wc[index]*(cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2)/np.sqrt(-4*N[index]**2*cs[index]**2*kxr**2 + (cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2)**2) + wc[index])
    dx_ds = cs[index]**2*kxr + 0.5*(-4*N[index]**2*cs[index]**2*kxr + 2*cs[index]**2*kxr*(cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2))/np.sqrt(-4*N[index]**2*cs[index]**2*kxr**2 + (cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2)**2)
    dz_ds = 1.0*cs[index]**2*kzr*(cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2)/np.sqrt(-4*N[index]**2*cs[index]**2*kxr**2 + (cs[index]**2*(kxr**2 + kzr**2) + wc[index]**2)**2) + cs[index]**2*kzr
    return np.array([dx_ds,dz_ds,dkx_ds,dkz_ds])

#----- Main Program -----#
# Extracting the data
datos = open('model_jcd.dat')
dat=[]
for lin in datos:
    l=lin.split()
    dat.append(l)
datos.close()

z=[]
P=[]
rho=[]
T=[]
for i in range(len(dat)-1):
    z.append(float(dat[i+1][0])) 
    P.append(float(dat[i+1][1]))  
    rho.append(float(dat[i+1][2]))   
    T.append(float(dat[i+1][3]))  
    
z=np.array(z)
P=np.array(P)
rho=np.array(rho)
T=np.array(T)

# Plotting the data
plt.figure(1)
plt.plot(z,P,)
plt.xlabel('Z [km]')
plt.ylabel('P [dyn/cm^2]')
plt.grid()
#plt.savefig('1.eps',format='eps')

plt.figure(2)
plt.plot(z,rho)
plt.xlabel('Z [km]')
plt.ylabel('rho [g/cm^2]')
plt.grid()
#plt.savefig('2.eps',format='eps')

plt.figure(3)
plt.plot(z,T)
plt.xlabel('Z [km]')
plt.ylabel('T [K]')
plt.grid()
#plt.savefig('3.eps',format='eps')


# Calculating cs, N, wc
gamma=5./3.
cs=np.sqrt(gamma*P/rho)
g=274*100 #cm/s^2
H=P/(rho*g)
N=np.sqrt((g*(gamma-1))/(H*gamma))
wc=cs/(2*H)

# Plotting cs, N, wc
plt.figure(4)
plt.plot(z,cs)
plt.xlabel('Z [km]')
plt.ylabel('cs [cm/s]')
plt.grid()
#plt.savefig('4.eps',format='eps')


plt.figure(5)
plt.plot(z,wc, label=r'$\omega_{c}$')
plt.plot(z,N, label='N')
plt.legend()
plt.xlabel('Z [km]')
plt.ylabel('Frecuencias [s^-1]')
plt.grid()
#plt.savefig('5.jpg',format='jpg')

textstr = '\n'.join((
	r'$\omega_{c}^{2}/N^{2}=%.2f$' % ((wc[1]/N[1])**2, ),

))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.15, 0.18, textstr, fontsize=10,transform=plt.gcf().transFigure, bbox = props)


# Calculating dcs, dN, dwc
paso=-np.abs(z[1]-z[2])
csz=np.gradient(cs,paso)
Nz=np.gradient(N,paso)
wcz=np.gradient(wc,paso)


#----- Section A -----#
# Different frequencies a fixed height

# Initial values
frec=np.array([2.0,3.0,3.5,5.0])*10**(-3) #Hz
omega=2*np.pi*frec
# the kx are fixed and are a constant, it comes from the integral, 
# therefore we define it at the reflection point for a fixed z

ind_r=500 # reflection index
kx=omega/cs[ind_r]
kz=0
x=0
t0=0.
tf=3.
n=1000

plt.figure(7)
initial=np.array([x,z[ind_r],kx[0],kz])
sol,t=rk4(problem, initial, t0, tf, n)
plt.plot(sol[0,:],sol[1,:],label=r'$\nu$=2.0 mHz')

initial=np.array([x,z[ind_r],kx[1],kz])
sol,t=rk4(problem, initial, t0, tf, n)
plt.plot(sol[0,:],sol[1,:],label=r'$\nu$=3.0 mHz')

initial=np.array([x,z[ind_r],kx[2],kz])
sol,t=rk4(problem, initial, t0, tf, n)
plt.plot(sol[0,:],sol[1,:],label=r'$\nu$=3.5 mHz')

initial=np.array([x,z[ind_r],kx[3],kz])
sol,t=rk4(problem, initial, t0, tf, n)
plt.plot(sol[0,:],sol[1,:],label=r'$\nu$=5.0 mHz')

plt.ylabel('Z [km]')
plt.xlabel('X [km]')

plt.xlim(0,100000)
plt.legend()
plt.grid()
#plt.savefig('osc1.eps',format='eps')


#----- Section B -----#
# Different frequencies a fixed height
# Initial values
frec=np.array([2.5])*10**(-3) #Hz
omega=2*np.pi*frec
kz=0
x=0
t0=0.
tf=3.
n=1000

ind_r=np.array([100,200,400,500]) reflection index
kx=omega/cs[ind_r]
zs=z[ind_r]

plt.figure(8)
initial=np.array([x,zs[0],kx[0],kz])
sol,t=rk4(problem, initial, t0, tf, n)
plt.plot(sol[0,:],sol[1,:],label=r'Z=%.2f Km'%zs[0])

initial=np.array([x,zs[1],kx[1],kz])
sol,t=rk4(problem, initial, t0, tf, n)
plt.plot(sol[0,:],sol[1,:],label=r'Z=%.2f Km'%zs[1])

initial=np.array([x,zs[2],kx[2],kz])
sol,t=rk4(problem, initial, t0, tf, n)
plt.plot(sol[0,:],sol[1,:],label=r'Z=%.2f Km'%zs[2])

initial=np.array([x,zs[3],kx[3],kz])
sol,t=rk4(problem, initial, t0, tf, n)
plt.plot(sol[0,:],sol[1,:],label=r'Z=%.2f Km'%zs[3])

plt.ylabel('Z [km]')
plt.xlabel('X [km]')

plt.xlim(0,50000)
plt.legend()
plt.grid()
#plt.savefig('osc2b.eps',format='eps')

