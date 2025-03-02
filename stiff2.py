import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

# Define the exact solution
def exact_solution(t, omega):
    return np.sin(omega * t) + np.exp(-20 * t)

# Define the differential equation
def tfunc(t, x, w):
    return -20 * x + 20 * np.sin(w * t) + w * np.cos(w * t)

# Runge-Kutta 4 method for a single step
def rungekutta4(func, t, h, x, w):
    k1 = h * func(t, x, w)
    k2 = h * func(t + h / 2, x + k1 / 2, w)
    k3 = h * func(t + h / 2, x + k2 / 2, w)
    k4 = h * func(t + h, x + k3, w)
    
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 

# Function to solve differential equation using vode
def solve_with_vode(method, x, t, t_max, dt, func, w):
    solver = ode(func)
    solver.set_integrator('vode', method=method)
    solver.set_initial_value(x, t)
    solver.set_f_params(w)

    # Arrays to store the solution
    t_values = [t]
    x_values = [x]

    # Integrate step by step
    while solver.successful() and solver.t < t_max:
        solver.integrate(solver.t + dt)
        t_values.append(solver.t)
        x_values.append(solver.y[0])

    return np.array(t_values), np.array(x_values)

if __name__ == '__main__':

    # Declare Variables
    x0 = 1
    t0 = 0
    step_sizes = [2, 1, 0.5, 0.1, 0.05, 0.02, 0.01]
    omega = 0.02
    dt_values = [2, 1, 0.1, 0.01]
    t_max = 20
    omega_values = [0.02, 0.2, 2, 20]
    error_RK4 = []
    grad = []
    
    # Compute Runge-Kutta approximations and exact solutions for various step sizes
    for h in step_sizes:
        x_rk4 = rungekutta4(tfunc, t0, h, x0, omega)
        x_exact = exact_solution(h, omega)
        
        print(f'Error RK4 for h={h}: {abs(x_rk4 - x_exact):.5e}')
        error_RK4.append(abs(x_rk4 - x_exact))

    # Compute the gradient in log-log graph
    log_h = np.log(step_sizes[1:])
    log_error = np.log(error_RK4[1:])

    gradient_RK4 = np.gradient(log_error, log_h)
    
    # Print the mean gradient (OC)
    mean_gradient = np.mean(gradient_RK4)
    print(f"Gradient of log(Error) vs. log(h): {mean_gradient:.2f}")

    # Plot results
    plt.figure()
    plt.loglog(step_sizes, error_RK4, '--', label="Error")
    plt.xlabel("Step Size (h)", fontsize=16)
    plt.ylabel("Absolute Error", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('rk4_error2.png')

    # Plot results
    plt.figure()
    error_RK4 = []
    for w in omega_values:
        for h in step_sizes:
            x_rk4 = rungekutta4(tfunc, t0, h, x0, w)
            x_exact = exact_solution(h, w)
            
            print(f'Error RK4 for h={h} and omega={w}: {abs(x_rk4 - x_exact):.5e}')
            error_RK4.append(abs(x_rk4 - x_exact))

        plt.loglog(step_sizes, error_RK4, '--', label=f"omega={w}")

        # Compute the gradient in log-log graph
        log_h = np.log(step_sizes[1:])
        log_error = np.log(error_RK4[1:])

        gradient_RK4 = np.gradient(log_error, log_h) 

        # Print the mean gradient
        mean_gradient = np.mean(gradient_RK4)
        print(f"Gradient of log(Error) vs. log(h) for w = {w}: {mean_gradient:.2f}")

        grad.append(mean_gradient)
        error_RK4 = []

    plt.xlabel("Step Size (h)", fontsize=16)
    plt.ylabel("Absolute Error", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('rk4_error_vary.png')

    # Plot results
    plt.figure()
    
    for dt in dt_values:

        # Solve using BDF method
        t_bdf, x_bdf = solve_with_vode('bdf', x0, t0, t_max, dt, tfunc, omega)

        # Solve using Adams method
        t_adams, x_adams = solve_with_vode('adams', x0, t0, t_max, dt, tfunc, omega)

        # Exact solution
        x_exact_bdf = exact_solution(t_bdf, omega)
        x_exact_adams = exact_solution(t_adams, omega)

        # Compute absolute errors
        error_bdf = abs(x_bdf - x_exact_bdf)
        error_adams = abs(x_adams - x_exact_adams)

        print(f'Error BDF for dt={dt}:')
        print(error_bdf)
        print(f'Error Adams for dt={dt}:')
        print(error_adams)

        plt.semilogy(t_bdf, error_bdf, '-', label=f"Error BDF (Stiff), time step: {dt}")
        plt.semilogy(t_adams, error_adams, '--', label=f"Error Adams (Non-Stiff), time step: {dt}")
    
    plt.xlabel("Time (t)", fontsize=16)
    plt.ylabel("Absolute Error", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('comparsion_error2.png')

    # Plot results for dt = 2
    plt.figure()
    dt = 2
    for w in omega_values:
        t_vals, x_vals = solve_with_vode('bdf', x0, t0, t_max, dt, tfunc, w)
        exact = exact_solution(t_vals, w)
        error = abs(x_vals - exact)

        print(f'Error BDF for dt={dt} and omega ={w}:')
        print(error)
        plt.semilogy(t_vals, error, label=f'BDF: w={w}')

    plt.xlabel("Time (t)", fontsize=16)
    plt.ylabel("Absolute Error", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('bdf_error.png')

    plt.figure()
    for w in omega_values:
        t_vals, x_vals = solve_with_vode('adams', x0, t0, t_max, dt, tfunc, w)
        exact = exact_solution(t_vals, w)
        error = abs(x_vals - exact)
        
        print(f'Error Adams for dt={dt} and omega ={w}:')
        print(error)
        plt.semilogy(t_vals, x_vals, label=f'Adams: w={w}')

    plt.xlabel("Time (t)", fontsize=16)
    plt.ylabel("Absolute Error", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('adams_error.png')

     # Plot results for dt = 2
    plt.figure()

    for w in omega_values:
        t_bdf, x_bdf = solve_with_vode('bdf', x0, t0, t_max, dt, tfunc, w)
        t_adams, x_adams = solve_with_vode('adams', x0, t0, t_max, dt, tfunc, w)


        exact_bdf = exact_solution(t_bdf, w)
        exact_adams = exact_solution(t_adams, w)

        error_bdf = abs(x_bdf - exact_bdf)
        error_adams = abs(x_adams - exact_adams)

        plt.semilogy(t_bdf, error_bdf, '-', label=f'BDF: w={w}')
        plt.semilogy(t_adams, error_adams, '--', label=f'Adams: w={w}')

    plt.xlabel("Time (t)", fontsize=16)
    plt.ylabel("Absolute Error", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('comparison_error3.png')