using OptimalControl, OrdinaryDiffEq, MINPACK, Plots

t0 = 0.0
tf = 1.0
x0 = -1.0
xf_fixed = 0.0 # target

# Maximised Hamiltonian
h(x, p) = p^2/4 - p*x # TO BE UPDATED

# Makes flow from Hamiltonian
f = Flow(Hamiltonian(h)) # (xf, pf) = f(t0, x0, p0, tf)

#println("Hamiltonian flow function: ", f((t0, tf), x0, 0.0)[1])
# Shooting function
function shoot!(s, p0)
    s[1] = f(t0 ,x0, p0,tf)[1] - xf_fixed # TO BE UPDATED
end

# Solve
p0_guess = 1.0 # initial guess
sol = fsolve(shoot!, [p0_guess])
p0 = sol.x[1]

# Plots
guess = f((t0, tf), x0, p0_guess)
fig1 = plot(guess, xlabel="t", label=[ "x (guess)" "p (guess)" ], linestyle=:dash)
sol = f((t0, tf), x0, p0)
plot!(fig1, sol, label=["x (sol)" "p (sol)"], linestyle=:solid)
plot!(fig1, [t0, tf], [xf_fixed[1], xf_fixed[1]], label="target", colour=:black)