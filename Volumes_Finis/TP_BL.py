#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: masssonr

Equation de Bukeley Leverett d_t S + d_x VT*( S^2/[S^2+(1-S)^2/rmu] ) = 0
modelise l'injection de gaz (CO2)dans un aquifÃ¨re saturÃ© en eau
S est la fraction volumique de gaz dans le rÃ©servoir et 1-S la fraction volumique d'eau  

Schema volume fini decentre amont 

On se ramene Ã  resoudre 

d_t S(t) = F(S(t)), 
S(0) = S0, 

Schema d'integration en temps Euler explicite et implicite 

avec la fct F ci dessous 

"""

import numpy as np
import matplotlib.pyplot as plt


#jeu de donnees

L=1000 # longueur du reservoir 

tf=3600*24*30*12*10  # temps final de simulation 

#rapport des viscosite de l'eau et du gaz muw/mug > 1
rmu = 10


#Vitesse en m/s  
VT = 1E-6

#nombre d'inconnues 
N=10

#pas du maillage 
h=L/N


#fonction flux scalaire f(s) = s**2/(s**2 + (1-s)**2/rmu)  
def f(s):
    z = VT*s**2/( s**2 + (1-s)**2/rmu )
    return z 

#derivee de la fonction flux scalaire f'(s)  
def fp(s):
    z = VT*2/rmu*s*(1-s)/(s**2 + (1-s)**2/rmu )**2   
    return z

#max de la derivee de la fonction flux scalaire f'(s)  sur [0,1]
def maxfp():  
    zz = np.linspace(0,1,1000)
    return np.max(fp(zz))     


#fonction F(S) vectorielle avec decentrage 
def F(S):
    V = np.zeros(N)
    for i in range(N):
        if i == 0:
            V[i]=(1/h)*(f(1)-f(S[0]))
        else : 
            V[i]=(1/h)*(f(S[i-1])-f(S[i]))
    return V


#derivee A = F'(S)
def DF(S):
    A = np.zeros((N,N))
    A[0, 0] = -(1/h) * fp(S[0])
    for i in range(1,N):
        A[i, i-1] = (1/h) * fp(S[i-1])
        A[i, i]   = -(1/h) * fp(S[i])
    return A 


X = np.linspace(h/2,L-h/2,N)

#schema Euler Explicite  
#y = x + dt*f(x)

def EulerExplicite(x,dt):
    return x + dt*F(x)

#schema Euler Implicite 
#on resoud G(y) = 0 avec G(y) = y-x - dt*F(y)

def EulerImplicite(x,dt):
    eps = 1.0e-6
    kmax = 100
    dsobj = 0.1
    y=x
    r = y - x - dt*F(y) 
    nr0 = np.linalg.norm(r)
    nr = nr0
    k=0
    while (nr/nr0>eps)and(k<kmax)and(nr>eps):
        J = np.eye(N) - dt * DF(y)
        dy = np.linalg.solve(J, -r)
        alpha = min(1,dsobj/np.linalg.norm(y))
        y = y + alpha*dy
        r = y - x - dt*F(y)
        nr = np.linalg.norm(r)
        k+=1
        
    return y


# BOUCLE EN TEMPS 
dtcfl1 =  h/maxfp()

dt0 = dtcfl1
t = 0
S = np.zeros(N)
while (t<tf):
    dt = min(dt0,tf-t)
    t = t + dt

    S = EulerImplicite(S,dt)
    
    plt.figure(1)
    plt.plot(X,S,'-b')
    
plt.figure(3)
plt.plot(X,S,'-b')
