import argparse
import numpy as np
import matplotlib.pyplot as plt

from NN_step2 import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Maxwell PINN prediction (no exact solution)."
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/maxwell_pinn-10000.pt",
        help="Checkpoint path (DeepXDE .pt file).",
    )

    parser.add_argument("--Nx", type=int, default=400, help="Number of x samples for plotting.")
    parser.add_argument("--L", type=float, default=6.0, help="Domain length (must match training).")
    parser.add_argument("--Tf", type=float, default=2.0, help="Final time used in training (must match).")

    # Choose times to plot (in the same units as training time)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--t",
        type=float,
        nargs="+",
        default=None,
        help="Times to plot, e.g. --t 0.0 0.2 0.4",
    )
    group.add_argument(
        "--tfrac",
        type=float,
        nargs="+",
        default=None,
        help="Times as fractions of Tf, e.g. --tfrac 0.0 0.25 0.5 (means t = frac*Tf).",
    )
    group.add_argument(
        "--tlin",
        type=int,
        default=None,
        help="Plot N times linearly spaced in [0, Tf], e.g. --tlin 6",
    )

    # Plot styling / options
    parser.add_argument(
        "--field",
        choices=["E", "H", "both"],
        default="both",
        help="Which field(s) to plot.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, save the figure to this path instead of showing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Build and restore model (must match training settings!) ----
    model = build_model(L=args.L, Tf=args.Tf)
    model.restore(args.ckpt)

    # ---- Choose times ----
    if args.t is not None:
        times = np.array(args.t, dtype=float)
    elif args.tfrac is not None:
        times = np.array(args.tfrac, dtype=float) * args.Tf
    elif args.tlin is not None:
        times = np.linspace(0.0, args.Tf, args.tlin)
    else:
        # default: a few snapshots
        times = np.array([0.0, 0.25 * args.Tf, 0.5 * args.Tf, 0.75 * args.Tf, args.Tf])

    # ---- x grid ----
    x_plot = np.linspace(0.0, args.L, args.Nx)

    # Helper: evaluate at fixed time t
    def predict_at_time(t):
        X = np.zeros((args.Nx, 2))
        X[:, 0] = x_plot
        X[:, 1] = t
        U = model.predict(X)  # shape (Nx, 2): [E, H]
        return U[:, 0], U[:, 1]

    # ---- Plot ----
    if args.field == "both":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axE, axH = axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for t in times:
        E, H = predict_at_time(t)

        if args.field == "E":
            ax.plot(x_plot, E, label=f"t={t:.3f}")
        elif args.field == "H":
            ax.plot(x_plot, H, label=f"t={t:.3f}")
        else:
            axE.plot(x_plot, E, label=f"t={t:.3f}")
            axH.plot(x_plot, H, label=f"t={t:.3f}")

    if args.field == "E":
        ax.set_title("E(x,t) snapshots")
        ax.set_xlabel("x")
        ax.set_ylabel("E")
        ax.grid(True)
        ax.legend()
    elif args.field == "H":
        ax.set_title("H(x,t) snapshots")
        ax.set_xlabel("x")
        ax.set_ylabel("H")
        ax.grid(True)
        ax.legend()
    else:
        axE.set_title("E(x,t) snapshots")
        axE.set_xlabel("x")
        axE.set_ylabel("E")
        axE.grid(True)
        axE.legend()

        axH.set_title("H(x,t) snapshots")
        axH.set_xlabel("x")
        axH.set_ylabel("H")
        axH.grid(True)
        axH.legend()

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"✅ Saved figure to: {args.save}")
    else:
        plt.show()

    # ---- Plot train/test loss curves ----
    try :
        prefix = args.ckpt.replace("-10000.pt", "")

        steps = np.load(prefix + "_steps.npy")
        loss_train = np.load(prefix + "_loss_train.npy")
        loss_test  = np.load(prefix + "_loss_test.npy")
        
        plt.figure()
        plt.semilogy(steps, loss_train, label="Train loss")
        plt.semilogy(steps, loss_test, label="Test loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Train vs Test loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError as e:
        print("⚠️ Could not find saved loss files.")
        print("Make sure NN_step2.py saves:")
        print("  <prefix>_steps.npy, <prefix>_loss_train.npy, <prefix>_loss_test.npy")
        print("Error:", e)

if __name__ == "__main__":
    main()
