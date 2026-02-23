# maxwell_pinn.py
import os
import deepxde as dde
import numpy as np

def build_model(
        m=1,
        c0=3e8,
        lr=1e-3,
        num_domain=2540*5,
        num_boundary=80,
        num_initial=160,
        num_test=2000,
        layer_size=None,
        activation="tanh",
        initializer="Glorot uniform",
):
    w = m * np.pi
    w_tilde = w * c0
    T = 2 * np.pi / w

    def sol_exacte(X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        E = np.sin(m * np.pi * x) * np.cos(w * t)
        H = np.cos(m * np.pi * x) * np.sin(w * t)
        return np.hstack((E, H))

    def E_init(X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        return np.sin(m * np.pi * x) * np.cos(w * t)

    def H_init(X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        return np.cos(m * np.pi * x) * np.sin(w * t)

    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 2*T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_E = dde.icbc.DirichletBC(geomtime,lambda X: np.zeros((len(X), 1)),
        lambda _, on_boundary: on_boundary,component=0,)

    ic_E = dde.icbc.IC(geomtime, E_init, lambda _, on_initial: on_initial, component=0)
    ic_H = dde.icbc.IC(geomtime, H_init, lambda _, on_initial: on_initial, component=1)

    def Maxwell_syst(x, y):
        e_r = 1.0
        mu_r = 1.0
        dE_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dE_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dH_dx = dde.grad.jacobian(y, x, i=1, j=0)
        dH_dt = dde.grad.jacobian(y, x, i=1, j=1)
        return [e_r * dE_dt - dH_dx, mu_r * dH_dt - dE_dx]

    data = dde.data.TimePDE(
        geomtime,
        Maxwell_syst,
        [bc_E, ic_E, ic_H],
        solution = sol_exacte,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
    )

    if layer_size is None:
        layer_size = [2] + [32] * 5 + [2]

    net = dde.nn.FNN(layer_size, activation, initializer)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr,metrics=['l2 relative error'])

    return model, sol_exacte, T


def train_and_save(
    ckpt_path="checkpoints/maxwell_pinn",
    iterations=40000,
    **build_kwargs,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    model, sol_exacte, T = build_model(**build_kwargs)
    losshistory, train_state = model.train(iterations=iterations)

    # Sauvegarde des poids
    model.save(ckpt_path)
    print(f"✅ Modèle sauvegardé : {ckpt_path}")

    # ---- SAVE HISTORY ----
    hist_path = ckpt_path + ".history.npz"
    np.savez(
        hist_path,
        steps=np.array(losshistory.steps),
        loss_train=np.array(losshistory.loss_train),
        loss_test=np.array(losshistory.loss_test) if losshistory.loss_test is not None else None,
        metrics_test=np.array(losshistory.metrics_test) if losshistory.metrics_test is not None else None,
    )
    print(f"✅ History saved: {hist_path}")

    return model, sol_exacte, T, losshistory


if __name__ == "__main__":
    # Entraînement uniquement quand on exécute ce fichier directement
    train_and_save(iterations=10000)
