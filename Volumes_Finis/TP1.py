#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:20:51 2024

@author: masssonr

resolution par la methode VF de l'equation elliptique 1D

- u''(x) = f(x) sur (0,L)
u(0) = uD
-u'(L) = g

"""

import numpy as np
import matplotlib.pyplot as plt

# longueur du domaine 
L=1.0 

# A COMPLETER 

def u(x):
    s = np.exp(np.sin(np.pi*x)) 
    return s

def up(x): 
    s = np.pi*np.cos(np.pi*x)*np.exp(np.sin(np.pi*x))
    return s 

def f(x):
    s = (np.pi**2)*(np.sin(np.pi*x)-np.cos(np.pi*x)**2)*np.exp(np.sin(np.pi*x))
    return s


uD = u(0) 
g = - up(L)


# calcul de X, Ah, Sh et Uh 

def VF(f,uD,g,L,N):

    h = L/N

    X=np.linspace(h/2, L-h/2,N)
    
 
    A=(2/h)*np.eye(N)+(-1/h)*np.eye(N,k=-1)+(-1/h)*np.eye(N,k=+1)
    A[0,0]+=1/h
    A[-1,-1]-=1/h

    Sh=np.array(h*f(X))
    Sh[0]+=2*uD/h
    Sh[-1]-=g
    

    Uh = np.linalg.solve(A,Sh)
  
    return X,Uh

########################




#nombre de mailles
#N= 5
N= 20
#N= 20

X,Uh = VF(f,uD,g,L,N)


#plot des solutions exactes et VF 
plt.figure(1)
plt.clf()
Xfine = np.linspace(0,L,200)
plt.plot(Xfine,u(Xfine), label="solution_exacte") 
plt.plot(X,Uh, label="solution_approchee")
plt.legend(loc="upper left")
plt.ylim(0.8, 3.5)
plt.show()


##########################


# etude de la convergence du schema fct de h = L/N 
Nmesh = 8
sizeh = np.zeros(Nmesh)
erreurl2 = np.zeros(Nmesh)
erreurh10 = np.zeros(Nmesh)

for imesh in range(Nmesh):

    N = 10*2**imesh

    h = L/N    
    sizeh[imesh] = h
    X,Uh = VF(f,uD,g,L,N)
    
    Erh = u(X) - Uh

# calcul de l'erreur l2 discrete
    erreur_2=0
    for i in range(N):
        erreur_2 += h * Erh[i]**2 
    erl2 = np.sqrt(erreur_2)
    erreurl2[imesh] = erl2

# calcul de l'erreur h10 discrete
    erreur_h10 = Erh[0]**2/(h/2)
    for i in range(N-1):
        erreur_h10 += ((Erh[i]-Erh[i+1])**2)/(h)
    erh10 = np.sqrt(erreur_h10)
    erreurh10[imesh] = erh10
    
  
# plot des erreurs en echelle log log     
plt.figure(2)
plt.ylabel(" erreur l2 et h10 ")
plt.xlabel(" pas du maillage h ")
plt.loglog(sizeh,erreurl2,"-xb", label="erreur l2")    
plt.loglog(sizeh,erreurh10,"-xr", label="erreur h10")    
plt.legend(loc="upper left")

# calcul des pentes des courbes d'erreur en echelle log 
droite=np.polyfit(np.log(sizeh),np.log(erreurl2),1)
print("ordre de convergence l2",droite[0])

droite=np.polyfit(np.log(sizeh),np.log(erreurh10),1)
print("ordre de convergence h10",droite[0])







