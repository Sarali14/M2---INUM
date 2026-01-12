#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: masssonr


    Stratigraphic model with diffusive sediment transport 
    Modelize the formation of sedimentary basins at large space and time scales 

    Two lithology Model 


  d_t \int_0^h(x,t) ci(x,z,t)dz + div(cis(x,t) k_i grad(psi(b(x,t)))) = 0 on (0,Lx)x(0,tf), i=1,2

  h(x,0) = hinit(x)   on (0,Lx)

   cis(x,t)*ki*grad(psi).n = g0i for x=0,  grad(psi).n = 0 for x = Lx  

  b(x,t) = hsea(t) - h(x,t)


  d_t c_i(x,z,t) = 0 for all z<h(x,t)
      c_i(x,h(x,t),t) = cis(x,t) if d_t h(x,t) > 0
      c_i(x,z,0) = ci0(x,z) for z < hinit(x)


  Finite Volume discretization using TPFA fluxes on unstructured meshes and upwinding of the cis 

  The linear advection equation on each column x is integrated exactly on the moving 1D mesh using the discrete data 
   cis(x,t) and h(x,t)


  Ouputs: * h(x,t), b(x,t)

   
          * hs(x,t) = min_(t<=s<=tf) h(x,t) (sediment layers at each time t, taking into account erosions)

          * ci(x,z,tf)  concentrations in the basin at final time tf 



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
Newtmax = 50 # maximum number of Newton iterations 
eps=1.0e-6 #  stopping criteria 

# initial composition cinit of litho 1 assumed constant for z<hinit(x)
cinit = 0.5

# Kc/Km ratio of diffusion coefficients between continental and marine environments 
Km = 1
Kc = 10

Ki = np.zeros(2)
Ki[0] = 1 # K[0]/K[1] = ratio of diffusion coefficients of litho 1 and 2 
Ki[1] = 5

g0 = np.zeros(2) # flux au bord x=0 pour les lithologies 0 et 1 
g0[0] = -10
g0[1] = -10





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


def residual(h,b,cs,ncol,hcol,ccol,dt):
# inputs 
# dt 
# h(N), b(N), cs(N)    
# ncol[k]: number of cells in z: ]-infty,hcol(k,0)], ... ,[hcol(k,ncol[k]-2),hcol(k,ncol[k])-1]
# ccol(k,l), l=0,ncol[k]-1: concentration of litho 1 in the cell l = 0,...,ncol[k]-1
        
# compute the residuals in each cell
# R(k,i)  equation i = 1,2 in cell k=1,...,N 
# compute the inner fluxes fluxint(1:Nint) and the upwind cells upwind(1:Nint) at each inner face i=1,...,Nint    
    
    R = np.zeros([N,2])

    for k in range(N):  
        s1 = 0
        s2 = 0
        
        if h[k]>=hcol[k,ncol[k]-1]:     
            s1 = s1 + cs[k]*(h[k]-hcol[k,ncol[k]-1])   
            s2 = s2 + (1-cs[k])*(h[k]-hcol[k,ncol[k]-1])        
        else:
            l = ncol[k]-1
            while (l>=1)and( h[k] < hcol[k,l-1] ):             
                s1 = s1 + ccol[k,l]*(hcol[k,l-1]-hcol[k,l])   
                s2 = s2 + (1-ccol[k,l])*(hcol[k,l-1]-hcol[k,l])                         
                l = l-1          
      
            s1 = s1 + ccol[k,l]*(h[k]-hcol[k,l])
            s2 = s2 + (1-ccol[k,l])*(h[k]-hcol[k,l])               
    

        R[k,0] = s1*volume[k]/dt
        R[k,1] = s2*volume[k]/dt       
        

                     
# inner fluxes and upwind cells            
    fluxint = np.zeros(Nint)
    upwind = np.zeros(Nint,dtype=int)    
    for i in range(Nint):
        # A COMPLETER fluxint et upwind 
              
# inner fluxes
    for i in range(Nint):
        # A COMPLETER assemblage des flux interieurs       
        

# A COMPLETER assemblage des flux au bord x=0


    return R,fluxint,upwind




def Jacobian(h,b,cs,ncol,hcol,ccol,dt,fluxint,upwind):
# Jacobian of R wrt to h and cs
#
# numbering:     0, ..., 2*N-1 
# equation [k,0]-> 2*k, eq [k,1]-> 2*k+1
# unknown  h[k]: 2*k
# unknown cs[k]: 2*k+1 

    A=np.zeros([2*N,2*N])

# accumulation term in each cell 
    for k in range(N):    
        if (h[k]>=hcol[k,ncol[k]-1]): 

#    eqs k lithos 1 et 2 par rapport a hk      
          A[2*k,2*k]     = A[2*k,2*k]    + cs[k]*volume[k]/dt
          A[2*k+1,2*k]   = A[2*k+1,2*k]  + (1-cs[k])*volume[k]/dt         

#    eqs k lithos 1 et 2 par rapport a csk         
          s = (h[k]-hcol[k,ncol[k]-1])*volume[k]/dt  
          A[2*k,2*k+1]     = A[2*k,2*k+1]    + s 
          A[2*k+1,2*k+1]   = A[2*k+1,2*k+1]  - s
      
        else:       
            l = ncol[k]-1
            while  (l>=1)and( h[k] < hcol[k,l-1] ):                 
                l = l-1          
      
#    eqs k lithos 1 et 2 par rapport a hk      
            A[2*k,2*k]     = A[2*k,2*k]   + ccol[k,l]*volume[k]/dt   
            A[2*k+1,2*k]   = A[2*k+1,2*k] + (1-ccol[k,l])*volume[k]/dt                      
 

# inner fluxes
    for i in range(Nint):
        k1 = CellsbyFaceInt[i,0]
        k2 = CellsbyFaceInt[i,1]
        kup = upwind[i] 
  
        psipk1 = - f_psip(b[k1])
        psipk2 = - f_psip(b[k2])
  
  
        # A COMPLETER            

    return A



#data structure for the uniform 1D mesh of size dx of the domain (0,Lx)
# cells m = 0:N-1

X = np.linspace(dx/2,Lx-dx/2,N)
volume = dx*np.ones(N)
 

#Interior faces: i = 0:Nint-1
CellsbyFaceInt = np.zeros([Nint,2],dtype=int)
surfaceint = np.ones(Nint)
for i in range(Nint):
  CellsbyFaceInt[i,0] = i
  CellsbyFaceInt[i,1] = i+1

#Boundary
CellbyFaceBound = np.zeros(Nbound,dtype=int)
fbound = np.zeros(Nbound)
CellbyFaceBound[0] = 0
CellbyFaceBound[1] = N-1 



#transmissibilities of interior faces
Tint = np.zeros(Nint)
for i in range(Nint):
    m1 = CellsbyFaceInt[i,0]
    m2 = CellsbyFaceInt[i,1]
    Tint[i] =  surfaceint[i]/np.abs(X[m2]-X[m1])



#  simulation 

#initialization 
h0 = f_hinit(X)
h = h0 

# init of the columns ccol,hcol,ncol 
# ncol(k): number of cells in z: ]-infty,hcol[k,0]], ... ,[hcol[k,ncol[k]-2),hcol[k,ncol[k]-1]]
# ccol[k,l], l=0,ncol[k]: concentration of litho 1 in the cell l = 0,...,ncol[k]-1

cs = cinit*np.ones(N)

ncol = np.ones(N,dtype=int)
ccol = np.zeros([N,ndt+1])
ccol[:,0] = cs
hcol = np.zeros([N,ndt+1])
hcol[:,0] = h0



t = 0

hs = np.zeros([N,ndt+1])
hs[:,0] = h0

plt.figure(1)
plt.title('h')
plt.plot (X,h0,'-r')  


for n in range(ndt):  # time loop 
    t = t + dt    
    hsea = f_hsea(t)
    
    h = h0 + 1e-3*dt #init Newton with sedimentation     
    
    b = np.ones(N)*hsea - h # bathymetry
    R,fluxint,upwind = residual(h,b,cs,ncol,hcol,ccol,dt) # initial newton residual 
    normR = np.linalg.norm(R)
    normR0 = normR

    itn = 1
    while ((normR/normR0>eps)and(itn<Newtmax)): #Newton loop 
        itn = itn + 1
        
        A = Jacobian(h,b,cs,ncol,hcol,ccol,dt,fluxint,upwind) #Jacobian matrix A = dR/dh(h)
                
        
        # correction of the Jacobian and residual 
        for k in range(N): 
            epsA = 1e-10
            if np.abs(A[2*k+1,2*k+1])<epsA :
                # csk indeterminee 
                #on garde la somme des equations de la maille k et on met l'equation 2 -> csk = 0        
                R[k,0] = R[k,0]+R[k,1]
                R[k,1] = 0
        
                A[2*k,:] = A[2*k,:] + A[2*k+1,:]
                A[2*k+1,:] = 0
                A[2*k+1,2*k+1] = 1            

        B = np.zeros(2*N)
        for k in range(N):
            B[2*k]   = -R[k,0]
            B[2*k+1] = -R[k,1]
       
        
        dU = np.linalg.solve(A,B) #solution
        
        dh = np.zeros(N)
        dcs = np.zeros(N)
        for k in range(N):
            dh[k] = dU[2*k]
            dcs[k] = dU[2*k+1]

        # A COMPLETER: relaxation du Newton 

        for k in range(N): # Newton update   
            h[k]  = h[k]  + dh[k]  
            cs[k] = cs[k] + dcs[k]
 
            if (cs[k]<0): # projection of cs on [0,1]
                cs[k] = 0
            if (cs[k]>1):
                cs[k] = 1
            

        b = np.ones(N)*hsea - h # bathymetry 
        R,fluxint,upwind  = residual(h,b,cs,ncol,hcol,ccol,dt) 
        normR = np.linalg.norm(R) 
        #print('norm R',normR)

    print('it newton',itn)
    
    if (itn>Newtmax-1): # if Newton not converged
        print('newton non converge',itn,normR)
        sys.exit()

    h0 = h 
    hs[:,n+1] = h    
 
    for k in range(N):
        if h[k]>=hcol[k,ncol[k]-1] :    
            ncol[k] = ncol[k] + 1
            hcol[k,ncol[k]-1] = h[k]
            ccol[k,ncol[k]-1] = cs[k]              
        else:       
            l = ncol[k]-1
            while (l>=1)and( h[k] < hcol[k,l-1] ):                    
                l = l-1          
           
            ncol[k] = l+1
            hcol[k,ncol[k]-1] = h[k]             
  
    
 #plot h 
    plt.figure(1)
    plt.title('h') 
    plt.plot (X,h,'-r')  
     
 
# computation of hs taking out erosions 
for j in range(ndt-1,-1,-1):
    for k in range(N):
        hs[k,j] =  np.min([hs[k,j],hs[k,j+1]])

plt.figure(2)
plt.title('hs')
for it in range(ndt+1):
    plt.plot(X,hs[:,it],'-b')


# paleo concentration de sable le long des verticales en k1,k2,k3
k1 = int(N/4);
k2 = int(N/2);
k3 = int(3*N/4);
   

#plot paleo concentration de sable c1(xki,z,tf) fct de z 
plt.figure(3)
plt.title(' paleo concentrations at wells ')
plt.plot(hcol[k1][0:ncol[k1]],ccol[k1][0:ncol[k1]],'-xr') 
plt.plot(hcol[k2][0:ncol[k2]],ccol[k2][0:ncol[k2]],'-xb') 
plt.plot(hcol[k3][0:ncol[k3]],ccol[k3][0:ncol[k3]],'-xm') 


