function trapezoid_implicit(t0, y0, tf, dt, f; max_iter=10, tol=1e-8)
    t = t0
    y = y0

    ts = [t]  # store time points
    ys = [y]  # store solution points

    while t < tf
        # Initial guess: use previous value
        y_new = y

        # Fixed-point iteration to solve implicit equation
        for k in 1:max_iter
            y_next = y + (dt / 2) * (f(t, y) + f(t + dt, y_new))
            if abs(y_next - y_new) < tol
                y_new = y_next
                break
            end
            y_new = y_next
        end

        # Update time and solution
        t += dt
        y = y_new

        # Store results
        push!(ts, t)
        push!(ys, y)
    end

    return ts, ys
end

function Euler_explicit(t0,y0,tf,dt,f)
	t=t0
	y=y0

	ts=[t]
	ys=[y]

	while t < tf
     
		y = y + dt * f(t, y)
        	t += dt

        
		push!(ts, t)
        	push!(ys, y)
    	end

    return ts, ys
end

function implicit_euler(t0, y0, tf, dt, f; max_iter=10, tol=1e-8)
    t = t0
    y = y0

    ts = [t]
    ys = [y]

    while t < tf
        t_next = t + dt

        # Initial guess: use previous value
        y_new = y

        # Fixed-point iteration to solve implicit equation
        for k in 1:max_iter
            y_next = y + dt * f(t_next, y_new)
            if abs(y_next - y_new) < tol
                y_new = y_next
                break
            end
            y_new = y_next
        end

        # Update solution
        t = t_next
        y = y_new

        push!(ts, t)
        push!(ys, y)
    end

    return ts, ys
end
