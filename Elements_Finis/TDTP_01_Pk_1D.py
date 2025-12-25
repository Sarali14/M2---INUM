import numpy as np

# ------------------------------------------------------------
# Fonctions de base P1 en 1D
# ------------------------------------------------------------

def Lambda_of_P1(x, x1, x2):
    return np.array([
        (x - x2) / (x1 - x2),
        (x - x1) / (x2 - x1)
    ])


def MGrad_Lambda_of_P1(x1, x2):
    return np.array([
        1.0 / (x1 - x2),
        1.0 / (x2 - x1)
    ])


# ------------------------------------------------------------
# Fonction second membre
# ------------------------------------------------------------
def Rhs_Function(x, rhs):
    if rhs == 1:
        return x
    elif rhs == 2:
        return x - 1
    else:
        return x


# ------------------------------------------------------------
# Fonctions de base et gradients en points de Gauss
# ------------------------------------------------------------
def Bf_et_Grad(Nve, xe1, xe2, Xsi):
    Ng = Xsi.shape[1]
    Grad_Xsi = MGrad_Lambda_of_P1(xe1, xe2)

    Xsi1 = Xsi[0, :]
    Xsi2 = Xsi[1, :]

    Bf = np.zeros((Nve, Ng))
    GradBf = np.zeros((Nve, Ng))

    if Nve == 2:  # P1
        Bf[0] = Xsi1
        Bf[1] = Xsi2

        GradBf[0] = Grad_Xsi[0]
        GradBf[1] = Grad_Xsi[1]

    elif Nve == 3:  # P2
        Bf[0] = Xsi1 * (2 * Xsi1 - 1)
        Bf[1] = Xsi2 * (2 * Xsi2 - 1)
        Bf[2] = 4.0 * Xsi1 * Xsi2

        GradBf[0] = Grad_Xsi[0] * (4 * Xsi1 - 1)
        GradBf[1] = Grad_Xsi[1] * (4 * Xsi2 - 1)
        GradBf[2] = -4 * Grad_Xsi[1] * (2 * Xsi2 - 1)

    elif Nve == 4:  # P3
        L1, L2 = Xsi1, Xsi2
        dL1, dL2 = Grad_Xsi

        Bf[0] = L1 * (3 * L1 - 1) * (3 * L1 - 2) / 2
        Bf[1] = L2 * (3 * L2 - 1) * (3 * L2 - 2) / 2
        Bf[2] = 9 * L1 * L2 * (3 * L1 - 2) / 2
        Bf[3] = 9 * L1 * L2 * (3 * L2 - 2) / 2

        GradBf[0] = (dL1 * (3 * L1 - 1) * (3 * L1 - 2) / 2
                     + L1 * (3 * dL1) * (3 * L1 - 2) / 2
                     + L1 * (3 * L1 - 1) * (3 * dL1) / 2)

        GradBf[1] = (dL2 * (3 * L2 - 1) * (3 * L2 - 2) / 2
                     + L2 * (3 * dL2) * (3 * L2 - 2) / 2
                     + L2 * (3 * L2 - 1) * (3 * dL2) / 2)

        GradBf[2] = (9 * dL1 * L2 * (3 * L1 - 1) / 2
                     + 9 * L1 * dL2 * (3 * L1 - 1) / 2
                     + 9 * L1 * L2 * (3 * dL1) / 2)

        GradBf[3] = (9 * dL1 * L2 * (3 * L2 - 1) / 2
                     + 9 * L1 * dL2 * (3 * L2 - 1) / 2
                     + 9 * L1 * L2 * (3 * dL2) / 2)

    else:  # Par défaut : P1
        Bf[0] = Xsi1
        Bf[1] = 1.0 - Xsi1
        GradBf[:] = Grad_Xsi[:, None]

    return Bf, GradBf


# ------------------------------------------------------------
# Points de Gauss 1D
# ------------------------------------------------------------
# ------------------------------------------------------------
# Points de Gauss 1D
# ------------------------------------------------------------
def IntegrationNum(Ngi):
    if Ngi == 1:
        W = np.array([1.0])
        X1 = np.array([0.5])

    elif Ngi == 2:
        W = np.array([0.5, 0.5])
        X1 = 0.5 * np.array([1 - 1 / np.sqrt(3), 1 + 1 / np.sqrt(3)])

    elif Ngi == 3:
        W = np.array([5 / 18, 8 / 18, 5 / 18])
        X1 = np.array([
            0.5 * (1 - np.sqrt(3 / 5)),
            0.5,
            0.5 * (1 + np.sqrt(3 / 5)),
        ])

    else:
        W = np.array([1/6, 4/6, 1/6])
        X1 = np.array([0.0, 0.5, 1.0])

    Xsi = np.zeros((2, len(X1)))
    Xsi[0] = X1
    Xsi[1] = 1 - X1

    return W, Xsi, len(W)


# ------------------------------------------------------------
# Matrice et vecteur local
# ------------------------------------------------------------
def Ae_et_Be_Num(NpG, Pk, xe1, xe2, Mu, Reac, rhs):
    W, Xsi, Ng = IntegrationNum(NpG)

    dx = abs(xe2 - xe1)
    Nve = Pk + 1

    Ae = np.zeros((Nve, Nve))
    Be = np.zeros((Nve, 1))

    Bf, GradBf = Bf_et_Grad(Nve, xe1, xe2, Xsi)

    for gp in range(Ng):
        xg = Xsi[0, gp] * xe1 + Xsi[1, gp] * xe2
        fg = Rhs_Function(xg, rhs)

        for i in range(Nve):
            Bi = Bf[i, gp]
            Gi = GradBf[i, gp]

            Be[i] += dx * W[gp] * fg * Bi

            for j in range(Nve):
                Bj = Bf[j, gp]
                Gj = GradBf[j, gp]
                Ae[i, j] += dx * W[gp] * (Mu * Gi * Gj + Reac * Bi * Bj)

    return Ae, Be


# ------------------------------------------------------------
# Exercice 2.1.1 – analytique
# ------------------------------------------------------------
def Exo_2_1_1(xe1, xe2):
    dx = abs(xe1 - xe2)

    Aex = np.array([
        [1/dx + dx/3,   -1/dx + dx/6],
        [-1/dx + dx/6,   1/dx + dx/3]
    ])

    Bex = np.array([
        [dx * (2 * xe1 + xe2) / 6],
        [dx * (xe1 + 2 * xe2) / 6]
    ])

    return Aex, Bex


# ------------------------------------------------------------
# Exercice 2.1.2 – analytique
# ------------------------------------------------------------
def Exo_2_1_2(xe1, xe2):
    dx = abs(xe1 - xe2)

    Aex = np.array([
        [ 7/(3*dx),  1/(3*dx), -8/(3*dx)],
        [ 1/(3*dx),  7/(3*dx), -8/(3*dx)],
        [-8/(3*dx), -8/(3*dx), 16/(3*dx)]
    ])

    Bex = np.array([
        [dx * (xe1 - 1) / 6],
        [dx * (xe2 - 1) / 6],
        [dx * (xe1 + xe2 - 2) / 3]
    ])

    return Aex, Bex


# ------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ------------------------------------------------------------

print("******************************************************")
print("Exo 2.1.1")

Pk = 1
xe1, xe2 = 2.0, 3.0
Mu = 1.0
alpha = 1.0
rhs = 1

for Np_Gauss in range(1, 4):
    Afe, Bfe = Ae_et_Be_Num(Np_Gauss, Pk, xe1, xe2, Mu, alpha, rhs)
    Aex, Bex = Exo_2_1_1(xe1, xe2)

    print(f"\nNombre de points de Gauss = {Np_Gauss}")
    print("Ae_analytique - Ae_numérique =")
    print(Aex - Afe)
    print("Be_analytique - Be_numérique =")
    print(Bex - Bfe)

print("******************************************************")
print("Exo 2.1.2")

Pk    = 2
alpha = 0.0
rhs   = 2

for Np_Gauss in range(1, 4):
    Afe, Bfe = Ae_et_Be_Num(Np_Gauss, Pk, xe1, xe2, Mu, alpha, rhs)
    Aex, Bex = Exo_2_1_2(xe1, xe2)

    print(f"\nNombre de points de Gauss = {Np_Gauss}")
    print("Ae_analytique - Ae_numérique =")
    print(Aex - Afe)
    print("Be_analytique - Be_numérique =")
    print(Bex - Bfe)
    print("------------------------------------------------------")
    