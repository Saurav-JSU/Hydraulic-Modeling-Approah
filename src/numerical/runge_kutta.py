"""
Runge-Kutta methods for solving differential equations in hydraulics.

This module provides implementations of various Runge-Kutta methods for
solving ordinary differential equations in hydraulic calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union

def runge_kutta_4(
    equation_func: Callable[[float, float], float],
    y_start: float,
    x_range: Tuple[float, float],
    n_steps: int
) -> Dict[str, np.ndarray]:
    """
    Solve an ODE using the classical fourth-order Runge-Kutta method.
    
    Parameters:
        equation_func (Callable): Function returning dy/dx at (x, y)
        y_start (float): Initial value of y
        x_range (Tuple[float, float]): (x_start, x_end)
        n_steps (int): Number of steps
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    x_start, x_end = x_range
    dx = (x_end - x_start) / n_steps
    
    # Initialize arrays
    x_values = np.linspace(x_start, x_end, n_steps + 1)
    y_values = np.zeros_like(x_values)
    y_values[0] = y_start
    
    # RK4 method
    for i in range(n_steps):
        x = x_values[i]
        y = y_values[i]
        
        # Calculate the four RK4 coefficients
        k1 = equation_func(x, y)
        k2 = equation_func(x + 0.5 * dx, y + 0.5 * dx * k1)
        k3 = equation_func(x + 0.5 * dx, y + 0.5 * dx * k2)
        k4 = equation_func(x + dx, y + dx * k3)
        
        # Update y
        y_values[i+1] = y + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return {
        'x': x_values,
        'y': y_values
    }

def adaptive_rk4(
    equation_func: Callable[[float, float], float],
    y_start: float,
    x_range: Tuple[float, float],
    tolerance: float = 1e-6,
    min_step: float = 1e-6,
    max_step: float = 0.1,
    max_steps: int = 10000
) -> Dict[str, np.ndarray]:
    """
    Solve an ODE using the adaptive step size fourth-order Runge-Kutta method.
    
    Parameters:
        equation_func (Callable): Function returning dy/dx at (x, y)
        y_start (float): Initial value of y
        x_range (Tuple[float, float]): (x_start, x_end)
        tolerance (float): Error tolerance per step
        min_step (float): Minimum allowed step size
        max_step (float): Maximum allowed step size
        max_steps (int): Maximum number of steps
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    x_start, x_end = x_range
    
    # Initialize arrays to store results
    x_values = [x_start]
    y_values = [y_start]
    
    # Initial step size
    dx = min(max_step, (x_end - x_start) / 10)
    
    x = x_start
    y = y_start
    
    step_count = 0
    
    while x < x_end and step_count < max_steps:
        # Adjust step size if we're near the end
        if x + dx > x_end:
            dx = x_end - x
        
        # Take a full step
        k1 = equation_func(x, y)
        k2 = equation_func(x + 0.5 * dx, y + 0.5 * dx * k1)
        k3 = equation_func(x + 0.5 * dx, y + 0.5 * dx * k2)
        k4 = equation_func(x + dx, y + dx * k3)
        
        y_full = y + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Take two half steps
        half_dx = dx / 2
        
        # First half step
        k1_half = equation_func(x, y)
        k2_half = equation_func(x + 0.5 * half_dx, y + 0.5 * half_dx * k1_half)
        k3_half = equation_func(x + 0.5 * half_dx, y + 0.5 * half_dx * k2_half)
        k4_half = equation_func(x + half_dx, y + half_dx * k3_half)
        
        y_half = y + (half_dx / 6.0) * (k1_half + 2 * k2_half + 2 * k3_half + k4_half)
        
        # Second half step
        x_half = x + half_dx
        k1_half2 = equation_func(x_half, y_half)
        k2_half2 = equation_func(x_half + 0.5 * half_dx, y_half + 0.5 * half_dx * k1_half2)
        k3_half2 = equation_func(x_half + 0.5 * half_dx, y_half + 0.5 * half_dx * k2_half2)
        k4_half2 = equation_func(x_half + half_dx, y_half + half_dx * k3_half2)
        
        y_double = y_half + (half_dx / 6.0) * (k1_half2 + 2 * k2_half2 + 2 * k3_half2 + k4_half2)
        
        # Estimate error
        error = abs(y_double - y_full) / dx
        
        # Check if error is acceptable
        if error <= tolerance or dx <= min_step:
            # Accept the step
            x += dx
            y = y_double  # Use the more accurate result from two half steps
            
            # Store the results
            x_values.append(x)
            y_values.append(y)
            
            step_count += 1
        
        # Calculate new step size
        if error > 0:  # Avoid division by zero
            dx_new = 0.9 * dx * min(max(np.sqrt(tolerance / error), 0.3), 2.0)
        else:
            dx_new = 2.0 * dx
        
        # Limit step size
        dx = min(max(dx_new, min_step), max_step)
    
    return {
        'x': np.array(x_values),
        'y': np.array(y_values)
    }

def runge_kutta_fehlberg(
    equation_func: Callable[[float, float], float],
    y_start: float,
    x_range: Tuple[float, float],
    tolerance: float = 1e-6,
    min_step: float = 1e-6,
    max_step: float = 0.1,
    max_steps: int = 10000
) -> Dict[str, np.ndarray]:
    """
    Solve an ODE using the Runge-Kutta-Fehlberg method (RKF45).
    
    Parameters:
        equation_func (Callable): Function returning dy/dx at (x, y)
        y_start (float): Initial value of y
        x_range (Tuple[float, float]): (x_start, x_end)
        tolerance (float): Error tolerance per step
        min_step (float): Minimum allowed step size
        max_step (float): Maximum allowed step size
        max_steps (int): Maximum number of steps
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    x_start, x_end = x_range
    
    # Initialize arrays to store results
    x_values = [x_start]
    y_values = [y_start]
    
    # Initial step size
    dx = min(max_step, (x_end - x_start) / 10)
    
    x = x_start
    y = y_start
    
    step_count = 0
    
    # RKF45 coefficients
    a = [0, 1/4, 3/8, 12/13, 1, 1/2]
    b = [
        [0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    c4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
    c5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
    
    while x < x_end and step_count < max_steps:
        # Adjust step size if we're near the end
        if x + dx > x_end:
            dx = x_end - x
        
        # Calculate RKF45 stages
        k = np.zeros(6)
        k[0] = equation_func(x, y)
        
        for i in range(1, 6):
            y_temp = y
            for j in range(i):
                y_temp += dx * b[i][j] * k[j]
            k[i] = equation_func(x + a[i] * dx, y_temp)
        
        # Calculate 4th and 5th order solutions
        y4 = y
        y5 = y
        
        for i in range(6):
            y4 += dx * c4[i] * k[i]
            y5 += dx * c5[i] * k[i]
        
        # Estimate error
        error = abs(y5 - y4)
        
        # Check if error is acceptable
        if error <= tolerance * dx or dx <= min_step:
            # Accept the step
            x += dx
            y = y5  # Use the more accurate 5th order result
            
            # Store the results
            x_values.append(x)
            y_values.append(y)
            
            step_count += 1
        
        # Calculate new step size
        if error > 1e-15:  # Avoid division by zero
            dx_new = 0.84 * dx * (tolerance * dx / error)**0.25
        else:
            dx_new = 2.0 * dx
        
        # Limit step size
        dx = min(max(dx_new, min_step), max_step)
    
    return {
        'x': np.array(x_values),
        'y': np.array(y_values)
    }

def runge_kutta_system(
    equation_func: Callable[[float, np.ndarray], np.ndarray],
    y_start: np.ndarray,
    x_range: Tuple[float, float],
    n_steps: int
) -> Dict[str, np.ndarray]:
    """
    Solve a system of ODEs using the fourth-order Runge-Kutta method.
    
    Parameters:
        equation_func (Callable): Function returning dy/dx at (x, y)
        y_start (np.ndarray): Initial values of y vector
        x_range (Tuple[float, float]): (x_start, x_end)
        n_steps (int): Number of steps
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    x_start, x_end = x_range
    dx = (x_end - x_start) / n_steps
    
    # Initialize arrays
    x_values = np.linspace(x_start, x_end, n_steps + 1)
    n_vars = len(y_start)
    y_values = np.zeros((n_steps + 1, n_vars))
    y_values[0, :] = y_start
    
    # RK4 method for system
    for i in range(n_steps):
        x = x_values[i]
        y = y_values[i, :]
        
        # Calculate the four RK4 coefficients
        k1 = equation_func(x, y)
        k2 = equation_func(x + 0.5 * dx, y + 0.5 * dx * k1)
        k3 = equation_func(x + 0.5 * dx, y + 0.5 * dx * k2)
        k4 = equation_func(x + dx, y + dx * k3)
        
        # Update y
        y_values[i+1, :] = y + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return {
        'x': x_values,
        'y': y_values
    }

def adaptive_runge_kutta_system(
    equation_func: Callable[[float, np.ndarray], np.ndarray],
    y_start: np.ndarray,
    x_range: Tuple[float, float],
    tolerance: float = 1e-6,
    min_step: float = 1e-6,
    max_step: float = 0.1,
    max_steps: int = 10000
) -> Dict[str, np.ndarray]:
    """
    Solve a system of ODEs using the adaptive step size Runge-Kutta-Fehlberg method.
    
    Parameters:
        equation_func (Callable): Function returning dy/dx at (x, y)
        y_start (np.ndarray): Initial values of y vector
        x_range (Tuple[float, float]): (x_start, x_end)
        tolerance (float): Error tolerance per step
        min_step (float): Minimum allowed step size
        max_step (float): Maximum allowed step size
        max_steps (int): Maximum number of steps
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    x_start, x_end = x_range
    n_vars = len(y_start)
    
    # Initialize arrays to store results
    x_values = [x_start]
    y_values = [y_start.copy()]
    
    # Initial step size
    dx = min(max_step, (x_end - x_start) / 10)
    
    x = x_start
    y = y_start.copy()
    
    step_count = 0
    
    # RKF45 coefficients
    a = [0, 1/4, 3/8, 12/13, 1, 1/2]
    b = [
        [0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    c4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
    c5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
    
    while x < x_end and step_count < max_steps:
        # Adjust step size if we're near the end
        if x + dx > x_end:
            dx = x_end - x
        
        # Calculate RKF45 stages
        k = np.zeros((6, n_vars))
        k[0, :] = equation_func(x, y)
        
        for i in range(1, 6):
            y_temp = y.copy()
            for j in range(i):
                y_temp += dx * b[i][j] * k[j, :]
            k[i, :] = equation_func(x + a[i] * dx, y_temp)
        
        # Calculate 4th and 5th order solutions
        y4 = y.copy()
        y5 = y.copy()
        
        for i in range(6):
            y4 += dx * c4[i] * k[i, :]
            y5 += dx * c5[i] * k[i, :]
        
        # Estimate error (maximum error across all variables)
        error = np.max(np.abs(y5 - y4))
        
        # Check if error is acceptable
        if error <= tolerance * dx or dx <= min_step:
            # Accept the step
            x += dx
            y = y5.copy()  # Use the more accurate 5th order result
            
            # Store the results
            x_values.append(x)
            y_values.append(y.copy())
            
            step_count += 1
        
        # Calculate new step size
        if error > 1e-15:  # Avoid division by zero
            dx_new = 0.84 * dx * (tolerance * dx / error)**0.25
        else:
            dx_new = 2.0 * dx
        
        # Limit step size
        dx = min(max(dx_new, min_step), max_step)
    
    # Convert lists to numpy arrays
    x_array = np.array(x_values)
    y_array = np.array(y_values)
    
    return {
        'x': x_array,
        'y': y_array
    }

def solve_boundary_value_problem(
    equation_func: Callable[[float, float, float, float], float],
    boundary_conditions: Tuple[Tuple[float, float], Tuple[float, float]],
    x_range: Tuple[float, float],
    n_points: int,
    tolerance: float = 1e-6,
    max_iterations: int = 50
) -> Dict[str, np.ndarray]:
    """
    Solve a two-point boundary value problem using the shooting method.
    
    Parameters:
        equation_func (Callable): Function returning y'' = f(x, y, y')
        boundary_conditions (Tuple): ((x0, y0), (x1, y1))
        x_range (Tuple[float, float]): (x_start, x_end)
        n_points (int): Number of grid points
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations for shooting method
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    (x0, y0), (x1, y1) = boundary_conditions
    x_start, x_end = x_range
    
    if abs(x0 - x_start) > 1e-10 or abs(x1 - x_end) > 1e-10:
        raise ValueError("Boundary condition x-values must match x_range")
    
    # Function to convert second-order ODE to system of first-order ODEs
    def system_func(x, y_vec):
        # y_vec[0] = y, y_vec[1] = y'
        return np.array([y_vec[1], equation_func(x, y_vec[0], y_vec[1])])
    
    # Function to solve initial value problem
    def solve_ivp(slope_guess):
        y_vec_start = np.array([y0, slope_guess])
        
        result = runge_kutta_system(system_func, y_vec_start, x_range, n_points-1)
        return result
    
    # Initial guesses for the slope at x0
    slope_low = -100.0
    slope_high = 100.0
    
    # Solve with initial guesses
    result_low = solve_ivp(slope_low)
    result_high = solve_ivp(slope_high)
    
    y_end_low = result_low['y'][-1, 0]
    y_end_high = result_high['y'][-1, 0]
    
    # Check if target is within the bracketed range
    if (y_end_low - y1) * (y_end_high - y1) >= 0:
        raise ValueError("Could not bracket the solution. Try different slope guesses.")
    
    # Bisection method to find correct initial slope
    for _ in range(max_iterations):
        slope_mid = (slope_low + slope_high) / 2
        
        result_mid = solve_ivp(slope_mid)
        y_end_mid = result_mid['y'][-1, 0]
        
        if abs(y_end_mid - y1) < tolerance:
            # Converged
            return {
                'x': result_mid['x'],
                'y': result_mid['y'][:, 0],
                'y_prime': result_mid['y'][:, 1]
            }
        
        if (y_end_mid - y1) * (y_end_low - y1) < 0:
            slope_high = slope_mid
            y_end_high = y_end_mid
        else:
            slope_low = slope_mid
            y_end_low = y_end_mid
    
    # If we reached max iterations without convergence, return the best result
    return {
        'x': result_mid['x'],
        'y': result_mid['y'][:, 0],
        'y_prime': result_mid['y'][:, 1]
    }

def solve_unsteady_flow(
    flux_func: Callable[[float, float, float], Tuple[float, float]],
    initial_condition: Callable[[float], Tuple[float, float]],
    boundary_conditions: Callable[[float], Tuple[Optional[float], Optional[float]]],
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    nx: int,
    nt: int
) -> Dict[str, np.ndarray]:
    """
    Solve unsteady flow equations using the method of characteristics.
    
    Parameters:
        flux_func (Callable): Function returning (F1, F2) for flux terms
        initial_condition (Callable): Function returning (h, Q) at t=0 for given x
        boundary_conditions (Callable): Function returning (h_upstream, h_downstream) at time t
        x_range (Tuple[float, float]): (x_start, x_end)
        t_range (Tuple[float, float]): (t_start, t_end)
        nx (int): Number of spatial grid points
        nt (int): Number of time steps
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x, t, h, and Q arrays
    """
    x_start, x_end = x_range
    t_start, t_end = t_range
    
    # Create grid
    x = np.linspace(x_start, x_end, nx)
    t = np.linspace(t_start, t_end, nt)
    
    dx = (x_end - x_start) / (nx - 1)
    dt = (t_end - t_start) / (nt - 1)
    
    # Initialize solution arrays
    h = np.zeros((nt, nx))
    Q = np.zeros((nt, nx))
    
    # Set initial conditions
    for i in range(nx):
        h[0, i], Q[0, i] = initial_condition(x[i])
    
    # Apply boundary conditions
    h_up, h_down = boundary_conditions(t_start)
    if h_up is not None:
        h[0, 0] = h_up
    if h_down is not None:
        h[0, -1] = h_down
    
    # Time stepping
    for n in range(nt - 1):
        # Apply boundary conditions for current time step
        h_up, h_down = boundary_conditions(t[n])
        if h_up is not None:
            h[n, 0] = h_up
        if h_down is not None:
            h[n, -1] = h_down
        
        # Interior points
        for i in range(1, nx - 1):
            # Calculate fluxes at current time step
            F1_i, F2_i = flux_func(x[i], h[n, i], Q[n, i])
            F1_im1, F2_im1 = flux_func(x[i-1], h[n, i-1], Q[n, i-1])
            F1_ip1, F2_ip1 = flux_func(x[i+1], h[n, i+1], Q[n, i+1])
            
            # Calculate spatial derivatives using central differences
            dF1_dx = (F1_ip1 - F1_im1) / (2 * dx)
            dF2_dx = (F2_ip1 - F2_im1) / (2 * dx)
            
            # Update solution using explicit time stepping
            h[n+1, i] = h[n, i] - dt * dF1_dx
            Q[n+1, i] = Q[n, i] - dt * dF2_dx
        
        # Handle boundary points
        # For simplicity, we use first-order extrapolation
        if h_up is None:
            h[n+1, 0] = 2 * h[n+1, 1] - h[n+1, 2]
        if h_down is None:
            h[n+1, -1] = 2 * h[n+1, -2] - h[n+1, -3]
        
        # Extrapolate Q at boundaries
        Q[n+1, 0] = 2 * Q[n+1, 1] - Q[n+1, 2]
        Q[n+1, -1] = 2 * Q[n+1, -2] - Q[n+1, -3]
    
    return {
        'x': x,
        'time': t,
        'h': h,
        'Q': Q
    }