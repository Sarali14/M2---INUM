import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def u0(x):
    # x est un tableau -> condition vectorisée
    return (x > 0.25).astype(float)

def s1(N,M,a,tf,xf,u):
    t=0
    x=0
    dt=tf/(N-1)
    dx=xf/(M-1)
    x=np.linspace(0,xf,M)
    t=np.linspace(0,tf,N)
    U=np.zeros((N,M))
    U[0,:]=u(x)
    U[:,0]=0.0
    c=dt/dx
    for n in range(0,N-1):
        U[n+1, 0] = 0.0
        for i in range(1,M-1):
            U[n+1,i]=U[n,i]-a*c*(U[n,i+1]-U[n,i])
        U[n+1, M-1] = U[n, M-1]
    return U,x,t

def s2(N,M,a,tf,xf,u):
    t=0
    x=0
    dt=tf/(N-1)
    dx=xf/(M-1)
    x=np.linspace(0,xf,M)
    t=np.linspace(0,tf,N)
    U=np.zeros((N,M))
    U[0,:]=u(x)
    U[:,0]=0.0
    c=dt/dx
    for n in range(0,N-1):
        U[n+1, 0] = 0.0
        for i in range(1,M-1):
            U[n+1,i]=U[n,i]-(1/2)*a*c*(U[n,i+1]-U[n,i-1])
        U[n+1, M-1] = U[n, M-1]
    return U,x,t

def s3(N,M,a,tf,xf,u):
    t=0
    x=0
    dt=tf/(N-1)
    dx=xf/(M-1)
    x=np.linspace(0,xf,M)
    t=np.linspace(0,tf,N)
    U=np.zeros((N,M))
    U[0,:]=u(x)
    U[:,0]=0.0
    c=dt/dx
    for n in range(0,N-1):
        U[n+1, 0] = 0.0
        for i in range(1,M-1):
            U[n+1,i]=U[n,i]-a*c*(U[n,i]-U[n,i-1])
        U[n+1, M-1] = U[n, M-1]
    return U,x,t
xf=1.0
tf=2.0

a=1/4

N=200
M=400

U1,x,t = s1(N,M,a,tf,xf,u0)
U2,x,t = s2(N,M,a,tf,xf,u0)
U3,x,t = s3(N,M,a,tf,xf,u0)

X, T = np.meshgrid(x, t)

fig = plt.figure(figsize=(18,5))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

ax1.plot_surface(X, T, U1, cmap='viridis')
ax1.set_title("Schéma 1")
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.set_zlabel("u")

ax2.plot_surface(X, T, U2, cmap='viridis')
ax2.set_title("Schéma 2")
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax2.set_zlabel("u")

ax3.plot_surface(X, T, U3, cmap='viridis')
ax3.set_title("Schéma 3")
ax3.set_xlabel("x")
ax3.set_ylabel("t")
ax3.set_zlabel("u")

plt.tight_layout()
plt.show()

