"""
Python translation of the provided Scilab/Matlab-style FEM script.

IMPORTANT (as requested):
- This is ONLY a translation.
- It does NOT solve the linear system.
- It does NOT plot.
- It keeps the same structure + placeholders (the "A FAIRE" parts raise NotImplementedError).
"""

import numpy as np


# =============================================================
# Exact solution
# =============================================================
"""print("----------------------------------------------")
print("--->> function out = Exact(x,y,kx,ky)         ")
print("!!!!!!!!-------------------------------!!!!!!!")
print("!!!!!!!!-------------------------------!!!!!!!")
print("!!!!!!!! A FAIRE !!!A FAIRE !!!A FAIRE !!!!!!!")
print("!!!!!!  Définir la bonne solution exacte   !!!")
print("!!!      nescessaire pour bien évaluer     !!!")
print("!!!       les erreurs d interpolation      !!!")
print("!!!!!!!!-------------------------------!!!!!!!")
print("----------------------------------------------")"""


def Exact(x, y, kx, ky):
    # vectorized like Scilab: sin(x*kx).*cos(y*ky)
    return np.sin(x * kx) * np.cos(y * ky)


# =============================================================
# RHS f
# =============================================================
"""print("----------------------------------------------")
print("--->> function out = f(X,kx,ky)              ")
print("!!!!!!!!-------------------------------!!!!!!!")
print("!!!!!!!! A FAIRE !!!A FAIRE !!!A FAIRE !!!!!!!")
print("!!!!!!!! Définir le bon second membre  !!!!!!!")
print("!!!!!!!!-------------------------------!!!!!!!")
print("----------------------------------------------")"""


def f(X, kx, ky,bx,by,cx,cy,alpha):
    x = X[0]
    y = X[1]
    return (alpha + bx*kx * kx + by*ky * ky) * Exact(x, y, kx, ky)+cx*kx*np.cos(kx*x)*np.cos(ky*y)+cy*ky*np.sin(kx*x)*np.sin(ky*y)


# =============================================================
# Composite coefficients
# =============================================================
"""print("----------------------------------------------")
print("--->> function [Conv, Reac, Diff ] ...       ")
print("--->>                 = Composite_Mat(Xg)    ")
print("!!!!!!!!-------------------------------!!!!!!!")
print("!!!!!!!! A FAIRE !!!A FAIRE !!!A FAIRE !!!!!!!")
print("!! Définir les bons coéficients du problème !!")
print("!!!!!!!!-------------------------------!!!!!!!")
print("----------------------------------------------")"""


def Composite_Mat(Xg):
    Conv = np.array([0.0,0.0])
    Reac = 1.0
    Diff = np.diag([0.0,0.0],k=0)
    return Conv, Reac, Diff


# =============================================================
# Structured Pk triangular mesh on a rectangle
# =============================================================
"""print("----------------------------------------------")
print("--->> function [N_vertives, N_elements,    ...")
print("                Coor, Nu, LogP, LogE, NuVe]...")
print("          = Struct_Pk_Mesh(Nx, Ny, Lx, Ly,pk) ")
print("!!!!!!!!-------------------------------!!!!!!!")
print("!!!  ALERT ALERT ALERT ALERT ALERT ALERT   !!!")
print("!!!  Ne pas modifier le mailleur intégré    !!")
print("!!!!!!!!-------------------------------!!!!!!!")"""


def Struct_Pk_Mesh(Nx, Ny, Lx, Ly, pk):
    MNx = Nx + (pk - 1) * (Nx - 1)
    MNy = Ny + (pk - 1) * (Ny - 1)

    N_vertices = MNx * MNy
    N_elements = 2 * (Nx - 1) * (Ny - 1)

    dx = Lx / (MNx - 1)
    dy = Ly / (MNy - 1)

    Coor = np.zeros((2, N_vertices), dtype=float)
    LogP = np.zeros((N_vertices,), dtype=int)

    is_ = 0
    for i in range(1, MNx + 1):
        xi = (i - 1) * dx
        for j in range(1, MNy + 1):
            yj = (j - 1) * dy
            is_ += 1
            # store at python index is_-1
            Coor[:, is_ - 1] = np.array([xi, yj], dtype=float)

            # boundary tags (same logic)
            tag = 0
            if i == 1:
                tag = -1
            elif i == MNx:
                tag = -2
            elif j == 1:
                tag = -10
            elif j == MNy:
                tag = -20
            LogP[is_ - 1] = tag

    # local dof per element for Pk-Lagrange triangles
    Ndl = int((pk + 1) * (pk + 2) / 2)

    Nu = np.zeros((Ndl, N_elements), dtype=int)  # will store 1-based node ids like Scilab
    LogE = np.zeros((N_elements,), dtype=int)

    ie = 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            is0 = 1 + pk * (j - 1) + (i - 1) * pk * MNy
            js0 = is0 + pk * MNy
            ks0 = js0 + pk
            ps0 = is0 + pk

            # ---- bottom triangle: is -> js -> ps
            ie += 1
            eidx = ie - 1
            Nu[0:3, eidx] = np.array([is0, js0, ps0], dtype=int)

            il = 3

            # edge (is -> js): (1,0)
            for lx in range(1, pk):
                il += 1
                Nu[il - 1, eidx] = is0 + lx * MNy

            # edge (js -> ps): (-1,1)
            for lo in range(1, pk):
                il += 1
                Nu[il - 1, eidx] = js0 + lo - lo * MNy

            # edge (ps -> is): (0,-1)
            for ly in range(1, pk):
                il += 1
                Nu[il - 1, eidx] = ps0 - ly

            # interior dofs in bottom triangle
            for lx in range(1, pk - 1):
                for ly in range(1, lx + 1):
                    il += 1
                    Nu[il - 1, eidx] = is0 + lx * MNy + ly

            LogE[eidx] = 0

            # ---- top triangle: js -> ks -> ps
            ie += 1
            eidx = ie - 1
            Nu[0:3, eidx] = np.array([js0, ks0, ps0], dtype=int)

            il = 3

            # edge (js -> ks): (0,+1)
            for ly in range(1, pk):
                il += 1
                Nu[il - 1, eidx] = js0 + ly

            # edge (ks -> ps): ??? (same as Scilab: ks - lx*MNy)
            for lx in range(1, pk):
                il += 1
                Nu[il - 1, eidx] = ks0 - lx * MNy

            # edge (ps -> js): (-1,1)
            for lo in range(1, pk):
                il += 1
                Nu[il - 1, eidx] = ps0 - lo + lo * MNy

            # interior dofs in top triangle
            for ly in range(1, pk - 1):
                for lx in range(1, ly + 1):
                    il += 1
                    Nu[il - 1, eidx] = js0 - lx * MNy + ly + 1

            LogE[eidx] = 0

    # NuVe: subdivision of a Pk triangle into P1 triangles for plotting
    if pk == 1:
        NuVe = np.array([[1], [2], [3]], dtype=int)
    elif pk == 2:
        NuVe = np.array(
            [[1, 4, 4, 6],
             [4, 5, 2, 5],
             [6, 6, 5, 3]],
            dtype=int
        )
    elif pk == 3:
        NuVe = np.array(
            [[1, 4, 4, 5, 5, 9, 10, 10, 8],
             [4, 10, 5, 6, 2, 10, 7, 6, 7],
             [9, 9, 10, 10, 6, 8, 8, 7, 3]],
            dtype=int
        )
    else:
        raise ValueError("NuVe is only defined for pk=1,2,3 in the original script.")

    return N_vertices, N_elements, Coor, Nu, LogP, LogE, NuVe


# =============================================================
# Pk Lagrange basis functions Phi
# =============================================================
def Phi_Pk(lmd1, lmd2, pk):
    lmd3 = 1.0 - lmd1 - lmd2

    if pk == 1:
        Phi = np.zeros((3,), dtype=float)
        Phi[0] = lmd1
        Phi[1] = lmd2
        Phi[2] = lmd3
        return Phi

    if pk == 2:
        Phi = np.zeros((6,), dtype=float)
        Phi[0] = lmd1 * (2.0 * lmd1 - 1.0)
        Phi[1] = lmd2 * (2.0 * lmd2 - 1.0)
        Phi[2] = lmd3 * (2.0 * lmd3 - 1.0)
        Phi[3] = 4.0 * lmd1 * lmd2
        Phi[4] = 4.0 * lmd2 * lmd3
        Phi[5] = 4.0 * lmd3 * lmd1
        return Phi

    if pk == 3:
        Phi = np.zeros((10,), dtype=float)
        Phi[0]= lmd1*(3.0*lmd1-2)*(3.0*lmd1-1)/2.0
        Phi[1]= lmd2*(3.0*lmd2-2)*(3.0*lmd2-1)/2.0
        Phi[2]= lmd3*(3.0*lmd3-2)*(3.0*lmd3-1)/2.0
        Phi[3]= 9.0*lmd1*lmd2*(3*lmd1-1)/2.0
        Phi[4]= 9.0*lmd1*lmd2*(3*lmd2-1)/2.0
        Phi[5]= 9.0*lmd2*lmd3*(3*lmd2-1)/2.0
        Phi[6]= 9.0*lmd2*lmd3*(3*lmd3-1)/2.0
        Phi[7]= 9.0*lmd1*lmd3*(3*lmd3-1)/2.0
        Phi[8]= 9.0*lmd1*lmd3*(3*lmd1-1)/2.0
        Phi[9]= 27.0*lmd1*lmd2*lmd3
        return Phi

    raise NotImplementedError(f"A FAIRE: Phi_Pk for pk={pk} (as in the original).")


# =============================================================
# Gradients of Pk basis functions
# =============================================================
def GradPhi_Pk(lmd1, lmd2, G1, G2, pk):
    """
    G1, G2 are gradients of barycentric coordinates lambda1, lambda2 in physical space.
    In the Scilab code:
      lmd3 = 1 - lmd1 - lmd2
      G3   = -G1 - G2
    """
    lmd3 = 1.0 - lmd1 - lmd2
    G3 = -np.asarray(G1) - np.asarray(G2)

    if pk == 1:
        GradPhi = np.zeros((3, 2), dtype=float)
        GradPhi[0, :] = G1
        GradPhi[1, :] = G2
        GradPhi[2, :] = G3
        return GradPhi

    elif pk == 2:
        GradPhi = np.zeros((6,2), dtype=float)

        GradPhi[0,:] = G1*(2.0*lmd1 - 1.0) + lmd1*(2.0*G1)
        GradPhi[1,:] = G2*(2.0*lmd2 - 1.0) + lmd2*(2.0*G2)
        GradPhi[2,:] = G3*(2.0*lmd3 - 1.0) + lmd3*(2.0*G3)

        GradPhi[3,:] = 4.0*(G1*lmd2 + lmd1*G2)
        GradPhi[4,:] = 4.0*(G2*lmd3 + lmd2*G3)
        GradPhi[5,:] = 4.0*(G3*lmd1 + lmd3*G1)

        return GradPhi

    elif pk == 3:
        GradPhi = np.zeros((10,2), dtype=float)
       
        GradPhi[0,:] = 0.5*(G1*(3.0*lmd1 - 2.0)*(3.0*lmd1 -1.0) + lmd1*(3.0*G1)*(3.0*lmd1 -1.0)+ lmd1*(3.0*lmd1 - 2.0)*(3.0*G1))
        GradPhi[1,:] = 0.5*(G2*(3.0*lmd2 - 2.0)*(3.0*lmd2 -1.0) + lmd2*(3.0*G2)*(3.0*lmd2 -1.0)+ lmd2*(3.0*lmd2 - 2.0)*(3.0*G2))
        GradPhi[2,:] = 0.5*(G3*(3.0*lmd3 - 2.0)*(3.0*lmd3 -1.0) + lmd3*(3.0*G3)*(3.0*lmd3 -1.0)+ lmd3*(3.0*lmd3 - 2.0)*(3.0*G3))

        GradPhi[3,:] = (9.0/2.0)*(G1*lmd2*(3.0*lmd1-1)+lmd1*G2*(3.0*lmd1-1)+lmd1*lmd2*(3.0*G1))
        GradPhi[4,:] = (9.0/2.0)*(G1*lmd2*(3.0*lmd2-1)+lmd1*G2*(3.0*lmd2-1)+lmd1*lmd2*(3.0*G2))
        GradPhi[5,:] = (9.0/2.0)*(G2*lmd3*(3.0*lmd2-1)+lmd2*G3*(3.0*lmd2-1)+lmd2*lmd3*(3.0*G2))
        GradPhi[6,:] = (9.0/2.0)*(G2*lmd3*(3.0*lmd3-1)+lmd2*G3*(3.0*lmd3-1)+lmd2*lmd3*(3.0*G3))
        GradPhi[7,:] = (9.0/2.0)*(G1*lmd3*(3.0*lmd3-1)+lmd1*G3*(3.0*lmd3-1)+lmd1*lmd3*(3.0*G3))
        GradPhi[8,:] = (9.0/2.0)*(G1*lmd3*(3.0*lmd1-1)+lmd1*G3*(3.0*lmd1-1)+lmd1*lmd3*(3.0*G1))
       
        GradPhi[9,:] = 27.0*(G1*lmd2*lmd3+lmd1*G2*lmd3+lmd1*lmd2*G3)
       
        return GradPhi

    raise NotImplementedError(f"A FAIRE: GradPhi_Pk for pk={pk} (as in the original).")


# =============================================================
# 2D determinant
# =============================================================
def Determinant(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return p[0] * q[1] - p[1] * q[0]


# =============================================================
# Barycentric coordinates in triangle (Xi, Xj, Xk)
# =============================================================
def Lambda_of_P1(Xc, Xi, Xj, Xk):
    Xc = np.asarray(Xc)
    Xi = np.asarray(Xi)
    Xj = np.asarray(Xj)
    Xk = np.asarray(Xk)

    Lam = np.zeros((3,), dtype=float)
    Lam[0] = Determinant(Xc - Xj, Xk - Xj) / Determinant(Xi - Xj, Xk - Xj)
    Lam[1] = Determinant(Xc - Xk, Xi - Xk) / Determinant(Xj - Xk, Xi - Xk)
    Lam[2] = Determinant(Xc - Xi, Xj - Xi) / Determinant(Xk - Xi, Xj - Xi)
    return Lam


def MGrad_Lambda_of_P1(Xi, Xj, Xk):
    Xi = np.asarray(Xi)
    Xj = np.asarray(Xj)
    Xk = np.asarray(Xk)

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    Grad = np.zeros((3, 2), dtype=float)

    Grad[0, 0] = Determinant(e1, Xk - Xj) / Determinant(Xi - Xj, Xk - Xj)
    Grad[0, 1] = Determinant(e2, Xk - Xj) / Determinant(Xi - Xj, Xk - Xj)

    Grad[1, 0] = Determinant(e1, Xi - Xk) / Determinant(Xj - Xk, Xi - Xk)
    Grad[1, 1] = Determinant(e2, Xi - Xk) / Determinant(Xj - Xk, Xi - Xk)

    Grad[2, 0] = Determinant(e1, Xj - Xi) / Determinant(Xk - Xi, Xj - Xi)
    Grad[2, 1] = Determinant(e2, Xj - Xi) / Determinant(Xk - Xi, Xj - Xi)

    return Grad


# =============================================================
# Numerical integration data on reference triangle
# =============================================================
def IntegrationNum(Ngi):
    """
    Returns:
      Poid: weights, shape (Ngo,)
      Xsi:  barycentric coordinates of gauss points, shape (3, Ngo)
      Ngo:  number of gauss points
    """
   

    if Ngi == 1:
        Ngo = Ngi
       
        Poid = np.zeros((Ngo,), dtype=float)
        Xsi = np.zeros((3, Ngo), dtype=float)
        Poid[0]=0.5
       
        Xsi[0,:]= 1.0/3.0
        Xsi[1,:]= 1.0/3.0
        Xsi[2,:]= 1 -  Xsi[0, :] - Xsi[1, :]
   
    elif Ngi == 3:
        Ngo = Ngi
       
        Poid = np.zeros((Ngo,), dtype=float)
        Xsi = np.zeros((3, Ngo), dtype=float)
        Poid[:]=1.0/6.0
       
        Xsi[0,:]= np.array([2.0/3.0, 1.0/6.0, 1.0/6.0])
        Xsi[1,:]= np.array([1.0/6.0, 4.0/6.0, 1.0/6.0])
        Xsi[2,:]= 1 -  Xsi[0, :] - Xsi[1, :]
       
    elif Ngi == 4:
        Ngo = Ngi
       
        Poid = np.zeros((Ngo,), dtype=float)
        Xsi = np.zeros((3, Ngo), dtype=float)
        Poid[0]=-27.0/96.0
        Poid[1:]=25.0/96.0
       
        Xsi[0,:]= np.array([1.0/3.0, 3.0/5.0, 1.0/5.0, 1.0/5.0])
        Xsi[1,:]= np.array([1.0/3.0, 1.0/5.0, 3.0/5.0, 1.0/5.0])
        Xsi[2,:]= 1 -  Xsi[0, :] - Xsi[1, :]
    # Else branch: "six points de Gauss" (hardcoded)
    else :
        Ngo = 6
       
        Poid = np.zeros((Ngo,), dtype=float)
        Xsi = np.zeros((3, Ngo), dtype=float)

        s1 = 0.11169079483905
        s2 = 0.0549758718227661
        aa = 0.445948490915965
        bb = 0.091576213509771
   
       
        Poid[0:3] = s2
        Poid[3:6] = s1
   
       
        Xsi[0, :] = np.array([bb, 1 - 2 * bb, bb, aa, aa, 1 - 2 * aa])
        Xsi[1, :] = np.array([bb, bb, 1 - 2 * bb, 1 - 2 * aa, aa, aa])
        Xsi[2, :] = 1.0 - Xsi[0, :] - Xsi[1, :]

    return Poid, Xsi, Ngo


# =============================================================
# Main script (translated) — with solving/plotting DISABLED
# =============================================================
def main():
    # ------------------------------------------
    # Gauss points and weights in reference element
    # ------------------------------------------
    Ngi = 60  # in the original, but IntegrationNum(60) goes to the "else" case -> 6 points

    # FEM choice
    EF_Pk = 1  # 1=P1, 2=P2, 3=P3

    # mesh params
    Lx, Ly = 1.0, 1.0
    Nx0, Ny0 = 10, 10  # unused in original loop (kept here)

    kx = np.pi / Lx
    ky = 2 * np.pi / Ly

    W, Lambda, Ngp = IntegrationNum(Ngi)

    # Convergence analysis
    Nconv = 5
    Vconv = np.array([[13, 25, 31, 61, 73, 91],
            [7, 13, 16, 31, 37, 46],
            [5, 9, 11, 21, 25, 31],],dtype=int,)

    # storage
    H = np.zeros((Nconv,), dtype=float)
    NddLC = np.zeros((Nconv,), dtype=int)
    VNx = np.zeros((Nconv,), dtype=int)
    ErrL1 = np.zeros((Nconv,), dtype=float)
    ErrL2 = np.zeros((Nconv,), dtype=float)
    ErrLinf = np.zeros((Nconv,), dtype=float)

    for rf in range(1, Nconv + 1):
        sf = rf
        Nx = int(Vconv[EF_Pk - 1, sf - 1])
        Ny = Nx

        # Build mesh
        Pk_Np, Pk_Ne, Pk_Coor, Pk_Nu, Pk_LogP, Pk_LogE, Pk_NuV = Struct_Pk_Mesh(Nx, Ny, Lx, Ly, EF_Pk)

        Pk_Npl = 3
        Pk_Ndl = int((EF_Pk + 1) * (EF_Pk + 2) / 2)

        # global dofs
        Nddl = Pk_Np
        Coor = Pk_Coor
        LogP = Pk_LogP

        # local
        Npl = Pk_Npl
        Ndl = Pk_Ndl
        Ne = Pk_Ne
        Nu = Pk_Nu

        # Initialize global matrix and RHS (assembled but NOT solved)
        MatA = np.zeros((Nddl, Nddl), dtype=float)
        VecB = np.zeros((Nddl,), dtype=float)

        # ==========================================================
        # Assembly
        # ==========================================================
        for e in range(1, Ne + 1):
            # NOTE: Nu stores 1-based ids, convert to 0-based for Coor access
            Xi = Coor[:, Nu[0, e - 1] - 1]
            Xj = Coor[:, Nu[1, e - 1] - 1]
            Xk = Coor[:, Nu[2, e - 1] - 1]

            DetE = abs(Determinant(Xj - Xi, Xk - Xi))

            for gp in range(1, Ngp + 1):
                Xg = (
                    Lambda[0, gp - 1] * Xi
                    + Lambda[1, gp - 1] * Xj
                    + Lambda[2, gp - 1] * Xk
                )
                wg = W[gp - 1]

                Coef_Conv, Coef_Reac, Coef_Diff = Composite_Mat(Xg)
                bx = Coef_Diff[0,0]
                by = Coef_Diff[-1,-1]
                cx = Coef_Conv[0]
                cy = Coef_Conv[-1]
                alpha = Coef_Reac
                GradP1 = MGrad_Lambda_of_P1(Xi, Xj, Xk)

                Phi = Phi_Pk(Lambda[0, gp - 1], Lambda[1, gp - 1], EF_Pk)
                GradPhi = GradPhi_Pk(
                    Lambda[0, gp - 1],
                    Lambda[1, gp - 1],
                    GradP1[0, :],
                    GradP1[1, :],
                    EF_Pk,
                )

                for k in range(1, Ndl + 1):
                    is_ = Nu[k - 1, e - 1]  # 1-based
                    Phi_is = Phi[k - 1]
                    GradPhi_is = GradPhi[k - 1, :]

                    Be_k = f(Xg, kx, ky,bx,by,cx,cy,alpha) * Phi_is
                    VecB[is_ - 1] += wg * DetE * Be_k

                    for kp in range(1, Ndl + 1):
                        js = Nu[kp - 1, e - 1]
                        Phi_js = Phi[kp-1]
                        GradPhi_js = GradPhi[kp - 1, :]

                        Ae_k_kp = (np.dot((Coef_Diff@GradPhi_is),GradPhi_js)  
                                   + Coef_Reac * Phi_is*Phi_js
                                   - Phi_is * np.dot(Coef_Conv,GradPhi_js))
                       
                        MatA[is_ - 1, js - 1] += wg * DetE * Ae_k_kp

        # ==========================================================
        # Boundary conditions
        # ==========================================================
        for is_ in range(1, Nddl + 1):
            Xg = Coor[:, is_ - 1]
            if LogP[is_ - 1] < 0:
                MatA[is_ - 1, :] = 0.0
                MatA[is_ - 1, is_ - 1] = 1.0
                VecB[is_ - 1] = Exact(Xg[0], Xg[1], kx, ky)

        # ==========================================================
        # SOLVE: disabled (requested)
        # ==========================================================
        VecSol = np.linalg.solve(MatA,VecB)  # would be the solution vector after solving

        # ==========================================================
        # Visualization + error computation: disabled (depends on VecSol)
        # ==========================================================
        h = Lx * Ly / Ne
        H[rf - 1] = np.sqrt(Lx * Ly / Ne)
        NddLC[rf - 1] = Nddl
        VNx[rf - 1] = Nx

        # In original code, ErrL1/L2/Linf are computed using VecSol -> disabled.
        ErrL1[rf - 1] = np.nan
        ErrL2[rf - 1] = np.nan
        ErrLinf[rf - 1] = np.nan

    # Convergence orders: disabled because ErrL* are nan without solving
    Ordre_de_Conv = None

    # Print what original prints (but without meaningful values)
    print(EF_Pk)
    print(["Nddl", "Norme L1", "Norme L2", "Norme Linf", "Erreur"])
    # original prints a matrix; here we show placeholders
    for rf in range(1, Nconv + 1):
        print([int(NddLC[rf - 1]), ErrL1[rf - 1], ErrL2[rf - 1], ErrLinf[rf - 1], ErrL2[rf - 1]])


if __name__ == "__main__":
    main()