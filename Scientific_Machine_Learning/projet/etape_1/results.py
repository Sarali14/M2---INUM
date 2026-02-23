import argparse
import numpy as np
import matplotlib.pyplot as plt
from NN_step1 import build_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Maxwell PINN results with selectable time window."
    )
    parser.add_argument(
        "--window",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=0,
        help=(
            "Choose time window to plot:\n"
            "  0: [0, T]\n"
            "  1: [T/2, 3T/2]\n"
            "  2: [T, 2T]\n"
            "  3: [3T/2, 5T/2]\n"
            "  4: [2T, 3T]\n"
            "  5: [0, 3T]"
        ),
    )
    parser.add_argument("--Nt", type=int, default=300, help="Number of time samples.")
    parser.add_argument("--Nx", type=int, default=300, help="Number of space samples.")
    parser.add_argument("--xval", type=float, default=0.3, help="x value for time plot.")
    parser.add_argument(
        "--tval_frac",
        type=float,
        default=0.25,
        help="Pick t_val as (T * tval_frac). Default 0.25 -> T/4.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/maxwell_pinn-10000.pt",
        help="Checkpoint path.",
    )
    return parser.parse_args()

def get_time_window(T, window_id: int):
    windows = [
        (0.0, 1.0),     # [0, T]
        (0.5, 1.5),     # [T/2, 3T/2]
        (1.0, 2.0),     # [T, 2T]
        (1.5, 2.5),     # [3T/2, 5T/2]
        (2.0, 3.0),     # [2T, 3T]
        (0.0, 3.0)      # [0, 3T]
    ]
    a, b = windows[window_id]
    return a * T, b * T

def main():
    args = parse_args()

    c0 = 3e8
    z0 = 120 * np.pi

    # Build the same model
    model, sol_exacte, T_model = build_model(m=1, lr=1e-3)
    model.restore(args.ckpt)

    hist_path = "checkpoints/maxwell_pinn.history.npz"
    hist = np.load(hist_path, allow_pickle=True)

    steps = hist["steps"]
    loss_train = hist["loss_train"]
    loss_test = hist["loss_test"]
    metrics_test = hist["metrics_test"]


    # IMPORTANT: keep a consistent definition of T for plotting.
    # Your build_model returns some T tied to the PDE setup; you were rescaling by 1/c0.
    # We'll do it once, and only use the rescaled T afterwards.
    T = T_model 

    Nx = args.Nx
    Nt = args.Nt
    x_plot = np.linspace(0, 1, Nx)

    # Choose time window for the time-plot
    t_min, t_max = get_time_window(T, args.window)
    t_plot = np.linspace(t_min, t_max, Nt)

    # Choose a single time for the space-plot (default: T/4, in the same time units as T)
    t_val = T*(1/c0) * args.tval_frac
    x_val = args.xval

    # Build evaluation grids
    X_plot = np.zeros((Nx, 2))
    X_plot[:, 0] = x_plot
    X_plot[:, 1] = t_val

    T_plot = np.zeros((Nt, 2))
    T_plot[:, 0] = x_val
    T_plot[:, 1] = t_plot

    # Predict
    U_exact_X = sol_exacte(X_plot)
    U_pred_X  = model.predict(X_plot)

    U_exact_T = sol_exacte(T_plot)
    U_pred_T  = model.predict(T_plot)

    E_exact_X, H_exact_X = U_exact_X[:, 0], U_exact_X[:, 1] / z0
    E_pred_X,  H_pred_X  = U_pred_X[:, 0],  U_pred_X[:, 1] / z0

    E_exact_T, H_exact_T = U_exact_T[:, 0], U_exact_T[:, 1] / z0
    E_pred_T,  H_pred_T  = U_pred_T[:, 0],  U_pred_T[:, 1] / z0

    # ---- Plot vs x at fixed t ----
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_plot, E_exact_X, label="E exact")
    plt.plot(x_plot, E_pred_X, "--", label="E PINN")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("E")
    plt.title(f"E à t={t_val:.3e}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_plot, H_exact_X, label="H exact")
    plt.plot(x_plot, H_pred_X, "--", label="H PINN")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("H")
    plt.title(f"H à t={t_val:.3e}")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ---- Plot vs t at fixed x ----
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_plot, E_exact_T, label=f"E exact pour x={x_val}")
    plt.plot(t_plot, E_pred_T, "--", label="E PINN")
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("E")
    plt.title(f"E sur t ∈ [{t_min:.3e}, {t_max:.3e}] à x={x_val}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_plot, H_exact_T, label=f"H exact pour x={x_val}")
    plt.plot(t_plot, H_pred_T, "--", label="H PINN")
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("H")
    plt.title(f"H sur t ∈ [{t_min:.3e}, {t_max:.3e}] à x={x_val}")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # -----------------------
    # Plot Train/Test loss + L2 error (from loaded history)
    # -----------------------
    train_loss = np.sum(loss_train, axis=1)
    
    plt.figure()
    plt.semilogy(steps, train_loss, label="Train Loss")
    
    if loss_test is not None and np.size(loss_test) > 0:
        test_loss = np.sum(loss_test, axis=1)
        plt.semilogy(steps, test_loss, label="Test Loss")
        
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train vs Test Loss")
        plt.show()
        
        if metrics_test is not None and np.size(metrics_test) > 0:
            l2_error = np.array(metrics_test).reshape(-1)
            plt.figure()
            plt.semilogy(steps, l2_error)
            plt.xlabel("Iterations")
            plt.ylabel("L2 Relative Error")
            plt.title("L2 Relative Error (Test)")
            plt.show()
    else:
        print("⚠️ metrics_test is empty in the history file.")

if __name__ == "__main__":
    main()
