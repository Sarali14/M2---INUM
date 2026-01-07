#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: masssonr

    Stratigraphic model with diffusive sediment transport 
    Modelize the formation of sedimentary basins at large space and time scales 

    Single lithology Model 


  d_t h(x,t) + div(grad(psi(b(x,t)))) = 0 on (0,Lx)x(0,tf)

  h(x,0) = hinit(x)   on (0,Lx)

  grad(psi).n = g0 for x=0,  grad(psi).n = g1 for x = Lx  

  b(x,t) = hsea(t) - h(x,t)


  Finite Volume discretization using TPFA fluxes on unstructured meshes 

  Ouputs: * h(x,t), b(x,t)

   
          * hs(x,t) = min_(t<=s<=tf) h(x,s) (sediment layers at each time t, taking into account erosions)



"""

import numpy as np
import matplotlib.pyplot as plt
import sys



#data
Lx=2
#space discretization
N=100
Nint = N-1
Nbound = 2
dx=Lx/N

#time discretization
tf = 1.5 # final simulation time
ndt = 50
dt = tf/ndt # initial time step

#Newton convergence 
Newtmax = 10 # maximum number of Newton iterations 
eps=1.0e-6 #  stopping criteria 


Km = 1
Kc = 10

g0 = -20
g1 = 0


def f_hsea(t):
    s = 25 + 5*np.cos(12*t) 
    return s

def f_psi(u):
    if (u<0):
        s = Kc*u
    else: 
        s = Km*u
    return s

def f_psip(u):
    if (u<0):
        s = Kc
    else: 
        s = Km
    return s


def f_hinit(x):
    s = 25*np.exp(-8*x/Lx) + 10
    return s


def residual(h,b,h0,dt):
    R = np.zeros(N)

    # A COMPLETER

    return R



def Jacobian(b,dt):


    A = np.zeros([N,N])

    # A COMPLETER
    
    return A 



#data structure for the uniform 1D mesh of size dx of the domain (0,Lx)
# cells m = 0:N-1

X = np.linspace(dx/2,Lx-dx/2,N)
volume = dx*np.ones(N)
 

#Interior faces: i = 0:Nint-1
CellsbyFaceInt = np.zeros([Nint,2],dtype=int)
# A COMPLETER surfaceint 
for i in range(Nint):
  # A COMPLETER  CellsbyFaceInt 
  

#Boundary
CellbyFaceBound = np.zeros(Nbound,dtype=int)
fbound = np.zeros(Nbound)
# A COMPLETER CellbyFaceBound et fbound 


#transmissibilities of interior faces
Tint = np.zeros(Nint)
for i in range(Nint):
    # A COMPLETER Tint 
    

#  simulation 

#initialization 
h0 = f_hinit(X)
h = h0 
t = 0

hs = np.zeros([N,ndt+1])
hs[:,0] = h0

plt.figure(1)
plt.title('h')
plt.plot (X,h0,'-r')  


for n in range(ndt):  # time loop 
    t = t + dt    
    hsea = f_hsea(t)
    
    b = np.ones(N)*hsea - h # bathymetry
    R = residual(h,b,h0,dt) # initial newton residual 
    normR = np.linalg.norm(R)
    normR0 = normR

    itn = 1
    while ((normR/normR0>eps)and(itn<Newtmax)): #Newton loop 
        itn = itn + 1
        A = Jacobian(b,dt) #Jacobian matrix A = dR/dh(h)
        dh = -np.linalg.solve(A,R) #solution dh 
        h = h + dh #Newton update         
        b = np.ones(N)*hsea - h
        R = residual(h,b,h0,dt) #residual  
        normR = np.linalg.norm(R) #residual norm  

    print('it newton',itn)
    if (itn>Newtmax-1): # if Newton not converged
        print('newton non converge',itn,normR)
        sys.exit()

    h0 = h 
    hs[:,n+1] = h    
 
 

 #plot h 
    plt.figure(1)
    plt.title('h') 
    plt.plot (X,h,'-r')  
     
 
# computation of hs taking out erosions 
for j in range(ndt-1,-1,-1):
    for k in range(N):
        # A COMPLETER hs 
        

# plt.figure(2)
# plt.title('hs')
# for it in range(ndt+1):
#     plt.plot(X,hs[:,it],'-b')






