import numpy as np
import matplotlib.pyplot as plt

def u0_2d(X, Y):
    # example: step in x like your 1D u0, replicated in y
    return (X > 0.25).astype(float)

def s1_2d(Nt, Nx, Ny, a, b, tf, xf, yf, u0):
    dt = tf/(Nt-1)
    dx = xf/(Nx-1)
    dy = yf/(Ny-1)

    x = np.linspace(0, xf, Nx)
    y = np.linspace(0, yf, Ny)
    t = np.linspace(0, tf, Nt)

    X, Y = np.meshgrid(x, y, indexing="ij")  # X[i,j], Y[i,j]

    U = np.zeros((Nt, Nx, Ny))
    U[0, :, :] = u0(X, Y)

    cx = a * dt/dx
    cy = b * dt/dy

    for n in range(Nt-1):
        # interior update
        U[n+1, 0:Nx-1, 0:Ny-1] = (
            U[n, 0:Nx-1, 0:Ny-1]
            - cx*(U[n, 1:Nx,   0:Ny-1] - U[n, 0:Nx-1, 0:Ny-1])
            - cy*(U[n, 0:Nx-1, 1:Ny  ] - U[n, 0:Nx-1, 0:Ny-1])
        )

        # boundary conditions (simple / like your 1D)
        # left boundary x=0 (inflow if a>0): set to 0
        U[n+1, 0, :] = 0.0
        # bottom boundary y=0 (inflow if b>0): set to 0
        U[n+1, :, 0] = 0.0

        # outflow boundaries: copy last interior value
        U[n+1, Nx-1, :] = U[n+1, Nx-2, :]
        U[n+1, :, Ny-1] = U[n+1, :, Ny-2]

    return U, x, y, t

# parameters
xf, yf, tf = 1.0, 1.0, 2.0
a, b = 1/4, 1/6
Nt, Nx, Ny = 200, 200, 200

U, x, y, t = s1_2d(Nt, Nx, Ny, a, b, tf, xf, yf, u0_2d)

# visualize a few time slices (2D heatmaps)
X, Y = np.meshgrid(x, y, indexing="ij")

fig, axes = plt.subplots(1, 3, figsize=(14,4))
for ax, k in zip(axes, [0, Nt//2, Nt-1]):
    im = ax.pcolormesh(X, Y, U[k], shading="auto")
    ax.set_title(f"t = {t[k]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
