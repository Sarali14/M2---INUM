import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import tf

m = 1 #parametre m - mode index
c0 = 3*1e8 #vitesse de la lumière
w = m * np.pi
w_tilde = w * c0
T = 2*np.pi/w_tilde

def sol_exacte(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    E = np.sin(m*np.pi*x) * np.cos(w*t)
    H = np.cos(m*np.pi*x) * np.sin(w*t)
    return np.hstack((E, H))

def cond_init(X):
    x = X[:, 0:1]
    t = X[:, 1:2]   # this will be 0 for on_initial points
    E = np.sin(m*np.pi*x) * np.cos(w*t)
    H = np.cos(m*np.pi*x) * np.sin(w*t)
    return np.hstack((E, H))

geom = dde.geometry.Interval(0.0,1.0) #definition de l'interval en espace
timedomain = dde.geometry.TimeDomain(0.0, T) #definition de l'interval en temps
geomtime = dde.geometry.GeometryXTime(geom, timedomain) #definition de l'interval [0,1]x[0,T]

bc = dde.icbc.DirichletBC(geomtime,
    lambda X: np.zeros((len(X), 2)),          # [E,H] = [0,0]
    lambda _, on_boundary: on_boundary)        # applies on x=0 and x=1


ic = dde.icbc.IC(geomtime, cond_init,lambda _, on_initial: on_initial) #cond initial 

def Maxwell_syst(x,y):
    e_r  = 1.0 #extraction du param e_r
    mu_r = 1.0 #extraction du param mu_r
    dy1_x = dde.grad.jacobian(y, x, i=0, j=0) #dE/dx
    dy1_t = dde.grad.jacobian(y, x, i=0, j=1) #dE/dt
    dy2_x = dde.grad.jacobian(y, x, i=1, j=0) #dH/dx
    dy2_t = dde.grad.jacobian(y, x, i=1, j=1) #dH/dt
    return [e_r * dy1_t - dy2_x ,mu_r * dy1_x - dy2_t] # sortie des 2 equations

data = dde.data.TimePDE(geomtime,Maxwell_syst, [bc, ic],num_domain=2540, num_boundary=80,num_initial=160)

layer_size = [2] + [20]*3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
