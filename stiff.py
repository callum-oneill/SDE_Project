import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import solve_ivp

# Define the exact solution
def exact_solution(t):
    return np.exp(-30 * t)

# Define the differential equation
def tfunc(t, x):
    return -30 * x

# Runge-Kutta 4 method for a single step
def rungekutta4(func, t, h, x):
    k1 = h * func(t, x)
    k2 = h * func(t + h / 2, x + k1 / 2)
    k3 = h * func(t + h / 2, x + k2 / 2)
    k4 = h * func(t + h, x + k3)
    
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 

# Function to solve differential equation using vode
def solve_with_vode(method, x, t, t_max, dt, func):
    solver = ode(func)
    solver.set_integrator('vode', method=method)
    solver.set_initial_value(x, t)

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
    dt_values = [2, 1, 0.1, 0.01]
    t_max = 20
    t_span = (0, t_max)
    t_eval = np.linspace(0, 10, 1000)

    # Compute Runge-Kutta approximations and exact solutions for various step sizes
    error_RK4 = []
    for h in step_sizes:
        x_rk4 = rungekutta4(tfunc, t0, h, x0)
        x_exact = exact_solution(h)
        
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
    plt.loglog(step_sizes, error_RK4, 'b--', label="Error")
    plt.xlabel("Step Size (h)", fontsize=16)
    plt.ylabel("Absolute Error", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('rk4_error.png')

    # Plot results
    plt.figure()

    for dt in dt_values:

        # Solve using BDF method
        t_bdf, x_bdf = solve_with_vode('bdf', x0, t0, t_max, dt, tfunc)

        # Solve using Adams method
        t_adams, x_adams = solve_with_vode('adams', x0, t0, t_max, dt, tfunc)

        # Exact solution
        x_exact_bdf = exact_solution(t_bdf)
        x_exact_adams = exact_solution(t_adams)

        # Compute  absolute errors
        error_bdf = abs(x_bdf - x_exact_bdf)
        error_adams = abs(x_adams - x_exact_adams)

        print(f'Error BDF for dt={dt}:')
        print(error_bdf)
        print(f'Error Adams for dt={dt}:')
        print(error_adams)

        plt.semilogy(t_bdf, error_bdf, '-', label=f"Error BDF (Stiff), time step: {dt}")
        plt.semilogy(t_adams, error_adams, '--', label=f"Error Adams (Non-Stiff), time step: {dt}")

    '''
    # Solve using LSODA
    sol_lsoda = solve_ivp(tfunc, t_span, [x0], method='LSODA', t_eval=t_eval)

    # Compute exact solution
    x_exact = exact_solution(t_eval)

    # Compute absolute error
    error_lsoda = np.abs(sol_lsoda.y[0] - x_exact)
    plt.semilogy(sol_lsoda.t, error_lsoda, 'r-', label="Error (LSODA)")
    
    '''
    plt.xlabel("Time (t)", fontsize=16)
    plt.ylabel("x(t)", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('comparison_error.png')