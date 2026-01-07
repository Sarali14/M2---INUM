"""
Traduction Scilab -> Python du code fourni (éléments finis 2D, Pk-Lagrange).
Auteur: conversion automatique
Dépendances: numpy
"""

import numpy as np

def MyDet2D(p, q):
    """Déterminant 2D de deux vecteurs p et q (p, q : array-like of length 2)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return p[0]*q[1] - p[1]*q[0]


def Lambda_of_P1(Xc, Xi, Xj, Xk):
    """Coordonnées barycentriques de Xc dans le triangle (Xi, Xj, Xk)."""
    Xc = np.asarray(Xc, dtype=float)
    Xi = np.asarray(Xi, dtype=float)
    Xj = np.asarray(Xj, dtype=float)
    Xk = np.asarray(Xk, dtype=float)

    denom1 = MyDet2D(Xi - Xj, Xk - Xj)
    denom2 = MyDet2D(Xj - Xk, Xi - Xk)
    denom3 = MyDet2D(Xk - Xi, Xj - Xi)

    L1 = MyDet2D(Xc - Xj, Xk - Xj) / denom1
    L2 = MyDet2D(Xc - Xk, Xi - Xk) / denom2
    L3 = MyDet2D(Xc - Xi, Xj - Xi) / denom3

    return np.array([L1, L2, L3], dtype=float)


def MGrad_Lambda_of_P1(Xi, Xj, Xk):
    """
    Gradient des coordonnées barycentriques (Grad lambda_i).
    Retourne un tableau (3,2) où chaque ligne i est le gradient de Lambda_i.
    """
    Xi = np.asarray(Xi, dtype=float)
    Xj = np.asarray(Xj, dtype=float)
    Xk = np.asarray(Xk, dtype=float)

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    denom_12 = MyDet2D(Xi - Xj, Xk - Xj)
    denom_23 = MyDet2D(Xj - Xk, Xi - Xk)
    denom_31 = MyDet2D(Xk - Xi, Xj - Xi)

    Grad = np.zeros((3,2), dtype=float)

    Grad[0,0] = MyDet2D(e1, Xk - Xj) / denom_12
    Grad[0,1] = MyDet2D(e2, Xk - Xj) / denom_12

    Grad[1,0] = MyDet2D(e1, Xi - Xk) / denom_23
    Grad[1,1] = MyDet2D(e2, Xi - Xk) / denom_23

    Grad[2,0] = MyDet2D(e1, Xj - Xi) / denom_31
    Grad[2,1] = MyDet2D(e2, Xj - Xi) / denom_31

    return Grad


def Phi_Pk(L1, L2, pk):
    """Fonctions de base Pk (L1,L2 sont scalaires). Retourne (Phi, Nve)."""
    L1 = float(L1)
    L2 = float(L2)
    L3 = 1.0 - L1 - L2

    if pk == 1:
        Phi = np.array([L1, L2, L3], dtype=float)
        Nve = 3
        return Phi, Nve

    elif pk == 2:
        Phi = np.zeros(6, dtype=float)
        Phi[0] = L1*(2.0*L1 - 1.0)
        Phi[1] = L2*(2.0*L2 - 1.0)
        Phi[2] = L3*(2.0*L3 - 1.0)
        Phi[3] = 4.0 * L1 * L2
        Phi[4] = 4.0 * L2 * L3
        Phi[5] = 4.0 * L3 * L1
        Nve = 6
        return Phi, Nve

    elif pk == 3:
        raise NotImplementedError("Pk = 3 non implémenté (comme dans le code Scilab).")
    else:
        raise NotImplementedError(f"Pk = {pk} non pris en charge.")


def Nddl_on_elemnt(pk):
    """Nombre de ddl (nœuds) par élément selon pk."""
    if pk == 1:
        return 3
    elif pk == 2:
        return 6
    elif pk == 3:
        return 10
    else:
        raise NotImplementedError(f"Nddl_on_elemnt: pk = {pk} non pris en charge.")


def GradPhi_Pk(L1, L2, G1, G2, pk):
    """
    Gradients des fonctions de base Pk.
    - L1,L2 scalaires
    - G1, G2 vecteurs (length 2) : gradients de lambda1 et lambda2
    Retourne GradPhi de taille (Nve, 2)
    """
    L1 = float(L1)
    L2 = float(L2)
    L3 = 1.0 - L1 - L2
    G1 = np.asarray(G1, dtype=float)
    G2 = np.asarray(G2, dtype=float)
    G3 = -G1 - G2

    if pk == 1:
        GradPhi = np.zeros((3,2), dtype=float)
        GradPhi[0,:] = G1
        GradPhi[1,:] = G2
        GradPhi[2,:] = G3
        return GradPhi

    elif pk == 2:
        GradPhi = np.zeros((6,2), dtype=float)

        # Phi1 = L1*(2L1 - 1) => grad = G1*(2L1-1) + L1*(2*G1)
        GradPhi[0,:] = G1*(2.0*L1 - 1.0) + L1*(2.0*G1)
        GradPhi[1,:] = G2*(2.0*L2 - 1.0) + L2*(2.0*G2)
        GradPhi[2,:] = G3*(2.0*L3 - 1.0) + L3*(2.0*G3)

        GradPhi[3,:] = 4.0*(G1*L2 + L1*G2)
        GradPhi[4,:] = 4.0*(G2*L3 + L2*G3)
        GradPhi[5,:] = 4.0*(G3*L1 + L3*G1)

        return GradPhi

    elif pk == 3:
        raise NotImplementedError("GradPhi pour pk=3 non implémenté.")
    else:
        raise NotImplementedError(f"GradPhi_Pk: pk = {pk} non pris en charge.")


def IntegrationNum(Ngi):
    """
    Points de Gauss et poids sur l'élément de référence (barycentriques).
    Si Ngi in {1,2,3} -> non implémenté (comme le Scilab).
    Sinon on renvoie la formule à 6 points.
    Retourne (Poid (W), Xsi (3,Ng), Ngo)
    """
    if Ngi in (1,2,3):
        raise NotImplementedError(f"IntegrationNum: cas Ngi = {Ngi} non implémenté.")
    else:
        Ngo = 6
        s1 = 0.11169079483905
        s2 = 0.0549758718227661
        aa = 0.445948490915965
        bb = 0.091576213509771

        Poid = np.empty(Ngo, dtype=float)
        Poid[0:3] = s2
        Poid[3:6] = s1

        Xsi = np.zeros((3, Ngo), dtype=float)
        Xsi[0, :] = np.array([bb, 1-2*bb, bb, aa, aa, 1-2*aa])
        Xsi[1, :] = np.array([bb, bb, 1-2*bb, 1-2*aa, aa, aa])
        Xsi[2, :] = 1.0 - Xsi[0,:] - Xsi[1,:]

        return Poid, Xsi, Ngo


def Myf(x):
    """Terme source f en un point x (x: [x,y])."""
    x = np.asarray(x, dtype=float)
    return np.sin(x[0]) * np.cos(x[1])


def Systeme_Local_Num(Xi, Xj, Xk, EF_Pk, Xsi, W, Ng):
    """
    Calcule la matrice locale Ae et le vecteur Be pour l'élément (Xi,Xj,Xk)
    EF_Pk : ordre Pk (1 ou 2)
    Xsi : tableau (3,Ng) des points barycentriques de Gauss
    W : tableau des poids (Ng,)
    Ng : nombre de points
    """
    Xi = np.asarray(Xi, dtype=float)
    Xj = np.asarray(Xj, dtype=float)
    Xk = np.asarray(Xk, dtype=float)

    DetE = abs(MyDet2D(Xj - Xi, Xk - Xi))
    Nve = Nddl_on_elemnt(EF_Pk)

    Ae = np.zeros((Nve, Nve), dtype=float)
    Be = np.zeros(Nve, dtype=float)

    # Boucle sur les points de Gauss
    for gp in range(Ng):
        l1 = Xsi[0, gp]
        l2 = Xsi[1, gp]
        l3 = Xsi[2, gp]

        Xg = l1*Xi + l2*Xj + l3*Xk
        wg = W[gp]

        GradP1 = MGrad_Lambda_of_P1(Xi, Xj, Xk)

        Phi, _ = Phi_Pk(l1, l2, EF_Pk)
        GradPhi = GradPhi_Pk(l1, l2, GradP1[0,:], GradP1[1,:], EF_Pk)

        # Assemblage local
        for k in range(Nve):
            Phi_k = Phi[k]
            GradPhi_k = GradPhi[k, :]
            Be_k = Myf(Xg) * Phi_k
            Be[k] += wg * DetE * Be_k

            for kp in range(Nve):
                Phi_kp = Phi[kp]
                GradPhi_kp = GradPhi[kp, :]
                Ae_k_kp = np.dot(GradPhi_k, GradPhi_kp) + Phi_k * Phi_kp
                Ae[k, kp] += wg * DetE * Ae_k_kp

    return Ae, Be


if __name__ == "__main__":
    # Paramètres
    NgCas = 11  # par défaut => on tombera dans le cas 6 points
    Wg, Xsig, Ng = IntegrationNum(NgCas)
    print(f"   Nombre de points de Gauss = {Ng}")

    # Exo 2.2.2 : triangle d'exemple
    Xe1 = np.array([0.0, 0.0])
    Xe2 = np.array([2.0, 0.0])
    Xe3 = np.array([0.0, 2.0])

    # Pk = 1
    Pk = 1
    Ae, Be = Systeme_Local_Num(Xe1, Xe2, Xe3, Pk, Xsig, Wg, Ng)
    print("------------------------------------------------------")
    print(f"    Eléments Finis Pk pour k = {Pk}")
    print("------------------------------------")
    print(" 6*Ae_numérique")
    print(6*Ae)
    print("Be_numérique")
    print(Be)
    print("------------------------------------------------------\n")

    # Pk = 2
    Pk = 2
    Ae, Be = Systeme_Local_Num(Xe1, Xe2, Xe3, Pk, Xsig, Wg, Ng)
    print("------------------------------------------------------")
    print(f"    Eléments Finis Pk pour k = {Pk}")
    print("------------------------------------")
    print("90*Ae_numérique")
    print(90*Ae)
    print("Be_numérique")
    print(Be)
    print("------------------------------------------------------")
