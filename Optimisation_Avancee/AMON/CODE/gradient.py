import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

ROOT = Path("/home/sarah-ali/M2---INUM/Master_2/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))

import windfarm_eval

Instance = str(ROOT / "instances/1/param3.txt")
X0_filz = str(ROOT / "instances/1/x3.txt")

def gradient_EAP(instance_path, X, h,l):
    """
    Compute the gradient of EAP w.r.t turbine positions using central finite differences.
    
    Parameters:
    - instance_path: path to the windfarm instance file
    - X: list of turbine positions [x0,y0,x1,y1,...]
    - h: finite difference step size
    
    Returns:
    - grad: list of derivatives w.r.t each x and y coordinate
    """
    # If X is a file path, read it
    if isinstance(X, str):
        with open(X, "r", encoding="utf-8") as f:
            s = f.read()
        s = s.replace("[", "").replace("]", "").replace(",", " ")
        X = [float(t) for t in s.split()]

    X = np.array(X,dtype=float)
    n_turbines = len(X)//2
    grad = [0.0] * len(X)
    
    for i in range(n_turbines):
        Xp = X.copy()
        Xm = X.copy()
        Yp = X.copy()
        Ym = X.copy()
        
        Xp[i*2] += h 
        Xm[i*2] -= h
        Yp[2*i+1] +=h
        Ym[2*i+1] -=h
    
        # Evaluate EAP at perturbed positions
        EAP_Xp,spacing_Xp,placing_Xp=windfarm_eval.windfarm_eval(instance_path, Xp.tolist())
        EAP_Xm,spacing_Xm,placing_Xm = windfarm_eval.windfarm_eval(instance_path, Xm.tolist())
        EAP_Yp,spacing_Yp,placing_Yp = windfarm_eval.windfarm_eval(instance_path, Yp.tolist())
        EAP_Ym,spacing_Ym,placing_Ym = windfarm_eval.windfarm_eval(instance_path, Ym.tolist())
        
        # Central difference approximation
        grad[2*i] =((EAP_Xp-l*(spacing_Xp+placing_Xp)) - (EAP_Xm-l*(spacing_Xm+placing_Xm))) / (2 * h)
        grad[2*i+1]=((EAP_Yp-l*(spacing_Yp+placing_Yp)) - (EAP_Ym-l*(spacing_Ym+placing_Ym))) / (2 * h)
    
    return grad
h=12
l=0.5
grad = gradient_EAP(Instance, X0_filz, h,l)
print("Gradient of EAP:", grad)
print("Norm of gradient:", np.linalg.norm(grad))


"""h_array = [1.0,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5]
grad_norms = []

for h in h_array:
    grad = gradient_EAP(Instance, X0_filz, h)
    grad_norms.append(np.linalg.norm(grad))

plt.plot(h_array, grad_norms, marker='o')
plt.xlabel("Step size h")
plt.ylabel("Gradient norm")
plt.title("Choosing optimal finite difference step")
plt.grid(True)
plt.show()"""


# --- Gradient descent ---
def gradient_descent(instance_path, X_init, h, alpha, tol, max_iter,l):
    X = np.array(X_init, dtype=float)
    it=0
    grad = gradient_EAP(instance_path, X, h,l)
    grad_norm = np.linalg.norm(grad)
    print(f"Initial gradient norm: {grad_norm}")

    while grad_norm > tol and it < max_iter:
        grad = np.array(gradient_EAP(instance_path, X, h,l), dtype=float)
        grad_norm = np.linalg.norm(grad)

        try:
            EAP_val, spacing,placing  = windfarm_eval.windfarm_eval(instance_path, X.tolist())
            EAP_val_penalise=EAP_val-l*(spacing+placing)
        except ValueError:
            print(f"Iteration {it+1}: Invalid turbine positions, reducing step size")
            alpha *= 0.5
            continue

        print(f"Iter {it+1}: EAP={EAP_val_penalise:.6f},alpha={alpha} ,Grad norm={grad_norm:.6f}")

        # Try a tentative update
        
        X_new = X + alpha * grad
        try:
            EAP_new,spacing,placing = windfarm_eval.windfarm_eval(instance_path, X_new.tolist())
            EAP_new_penalise=EAP_new-l*(spacing+placing)
        except ValueError:
            # Reduce alpha if new position is invalid
            alpha *= 0.5
            continue

        # Optional: reduce alpha if EAP did not increase
        if EAP_new_penalise < EAP_val_penalise:
            alpha *= 0.5
        else:
            X = X_new  # accept step

        it += 1

    return X.tolist()    

# --- Load initial turbine positions ---
with open(X0_filz, "r") as f:
    s = f.read().replace("[", "").replace("]", "").replace(",", " ")
X_init = [float(t) for t in s.split()]

# --- Run gradient descent ---
X_opt = gradient_descent(Instance, X_init, h=12, alpha=1000, tol=1e-3, max_iter=100,l=l)
print("Optimized positions:", X_opt)
