import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Domain and parameters
# ------------------------
Lx, Ly = 1.0, 1.0
kx = np.pi / Lx
ky = 2.0 * np.pi / Ly

# Exact solution
def exact(x, y, kx, ky):
    return np.sin(kx * x) * np.cos(ky * y)

# Grid
nx, ny = 101, 101
x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing="xy")   # shape (ny, nx)
U = exact(X, Y, kx, ky)

# ========================
# 3D surface plot
# ========================
fig1 = plt.figure(figsize=(7, 5))
ax1 = fig1.add_subplot(111, projection="3d")

surf = ax1.plot_surface(
    X, Y, U,
    cmap="jet",
    linewidth=0,
    antialiased=True
)

fig1.colorbar(surf, ax=ax1, shrink=0.6)
ax1.set_title("Exact solution — 3D surface")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u")

# ========================
# 2D colormap plot
# ========================
fig2, ax2 = plt.subplots(figsize=(6, 5))

cf = ax2.contourf(
    X, Y, U,
    levels=30,
    cmap="jet"
)

fig2.colorbar(cf, ax=ax2)
ax2.set_title("Exact solution — 2D colormap")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.tight_layout()
plt.show()
