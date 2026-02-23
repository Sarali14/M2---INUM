import os
import deepxde as dde
import numpy as np
import torch

def build_model(
        L=6.0,
        Tf=2.0,
        xg = 2.0,
        x_interface = 3.0, 
        alpha=10.0,
        c0=3e8,
        lr=1e-3,
        num_domain=2540,
        num_boundary=80,
        num_initial=160,
        num_test = 2000,
        layer_size=None,
        activation="tanh",
        initializer="Glorot uniform",
):
       
    def E_init(X):
        x = X[:, 0:1]
        return np.exp(-alpha * (x - xg) ** 2)

    def H_init(X):
        x = X[:, 0:1]
        return -np.exp(-alpha * (x - xg) ** 2)

    geom = dde.geometry.Interval(0.0, L)
    timedomain = dde.geometry.TimeDomain(0.0, Tf)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_E = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, component=0)
    bc_H = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, component=1)

    ic_E = dde.icbc.IC(geomtime, E_init, lambda _, on_initial: on_initial, component=0)
    ic_H = dde.icbc.IC(geomtime, H_init, lambda _, on_initial: on_initial, component=1)

    def Maxwell_syst(x, y):
        mu_r = 1.0
        xx = x[:, 0:1]
        
        e_r = torch.where(xx < x_interface, torch.ones_like(xx), 2.0 * torch.ones_like(xx))

        dE_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dE_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dH_dx = dde.grad.jacobian(y, x, i=1, j=0)
        dH_dt = dde.grad.jacobian(y, x, i=1, j=1)
        return [e_r * dE_dt - dH_dx, mu_r*dH_dt - dE_dx ]

    data = dde.data.TimePDE(
        geomtime,
        Maxwell_syst,
        [bc_E,bc_H,ic_E, ic_H],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
    )

    if layer_size is None:
        layer_size = [2] + [20] * 3 + [2]

    net = dde.nn.FNN(layer_size, activation, initializer)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    return model


def train_and_save(
    ckpt_path="checkpoints/maxwell_pinn",
    iterations=10000,
    **build_kwargs,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    model  = build_model(**build_kwargs)
    losshistory, train_state = model.train(iterations=iterations)

    steps = np.array(losshistory.steps)
    loss_train = np.sum(np.array(losshistory.loss_train), axis=1)
    loss_test  = np.sum(np.array(losshistory.loss_test), axis=1)
    
    np.save(ckpt_path + "_steps.npy", steps)
    np.save(ckpt_path + "_loss_train.npy", loss_train)
    np.save(ckpt_path + "_loss_test.npy", loss_test)

    # Sauvegarde des poids
    model.save(ckpt_path)
    print(f"✅ Modèle sauvegardé : {ckpt_path}")

    return model,  losshistory


if __name__ == "__main__":
    # Entraînement uniquement quand on exécute ce fichier directement
    train_and_save(iterations=10000)
