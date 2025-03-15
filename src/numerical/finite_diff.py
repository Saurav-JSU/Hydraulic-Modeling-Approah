"""
Finite difference methods for solving differential equations in hydraulics.

This module provides implementations of finite difference methods for
solving ordinary and partial differential equations in open channel hydraulics.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union

def direct_step_method(
    equation_func: Callable[[float], float],
    y_start: float,
    y_end: float,
    x_start: float = 0.0,
    max_iterations: int = 1000,
    min_step: float = 0.001,
    max_step: float = 10.0,
    tolerance: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Solve an ODE using the direct step method.
    
    Parameters:
        equation_func (Callable): Function returning dy/dx for current y
        y_start (float): Initial value of y
        y_end (float): Target value of y
        x_start (float): Initial value of x
        max_iterations (int): Maximum number of iterations
        min_step (float): Minimum step size for y
        max_step (float): Maximum step size for y
        tolerance (float): Convergence tolerance
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    if abs(y_start - y_end) < tolerance:
        return {
            'x': np.array([x_start]),
            'y': np.array([y_start])
        }
    
    # Direction of y changes
    direction = 1 if y_end > y_start else -1
    
    # Initialize arrays to store results
    x_values = [x_start]
    y_values = [y_start]
    
    current_y = y_start
    current_x = x_start
    
    # Main iteration loop
    for _ in range(max_iterations):
        # Calculate derivative dy/dx
        dy_dx = equation_func(current_y)
        
        # Check if derivative is too small (near horizontal)
        if abs(dy_dx) < 1e-10:
            # Use a small step in x instead
            if direction > 0:
                next_y = min(current_y + min_step, y_end)
            else:
                next_y = max(current_y - min_step, y_end)
                
            # Calculate dx for this change in y
            dx = (next_y - current_y) / 1e-6  # Use small dummy value
            
            # Limit step size
            if abs(dx) > max_step:
                dx = direction * max_step
                next_y = current_y + dx * 1e-6
        else:
            # Calculate step size for y
            if direction > 0:
                dy = min(max_step * abs(dy_dx), min_step / abs(dy_dx), y_end - current_y)
                dy = max(dy, min_step)
            else:
                dy = max(-max_step * abs(dy_dx), -min_step / abs(dy_dx), y_end - current_y)
                dy = min(dy, -min_step)
            
            # Calculate dx for this change in y
            dx = dy / dy_dx
            next_y = current_y + dy
        
        # Update current position
        current_x += dx
        current_y = next_y
        
        # Store values
        x_values.append(current_x)
        y_values.append(current_y)
        
        # Check if we've reached the target
        if abs(current_y - y_end) < tolerance:
            break
    
    return {
        'x': np.array(x_values),
        'y': np.array(y_values)
    }

def standard_step_method(
    x_values: np.ndarray,
    equation_func: Callable[[float, float], float],
    boundary_condition: Tuple[float, float],
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Solve an ODE using the standard step method with known x locations.
    
    Parameters:
        x_values (np.ndarray): Array of x locations to solve at
        equation_func (Callable): Function returning error for given (x, y)
        boundary_condition (Tuple[float, float]): (x, y) at boundary
        max_iterations (int): Maximum iterations per step
        tolerance (float): Convergence tolerance
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    # Sort x values to ensure they're in ascending order
    x_values = np.sort(x_values)
    n_points = len(x_values)
    
    # Initialize solution array
    y_values = np.zeros_like(x_values)
    
    # Find index of boundary location
    bc_x, bc_y = boundary_condition
    
    # Find closest x value to boundary condition
    bc_idx = np.argmin(np.abs(x_values - bc_x))
    y_values[bc_idx] = bc_y
    
    # Solve in upstream direction
    for i in range(bc_idx - 1, -1, -1):
        # Initial guess: use previous value
        y_guess = y_values[i+1]
        
        # Iterative solution for current location
        for iter_count in range(max_iterations):
            # Calculate error from equation
            error = equation_func(x_values[i], y_guess)
            
            if abs(error) < tolerance:
                break
                
            # Adjust guess using secant method if possible, otherwise use damped update
            # This is a simplified approach; a more robust method would use bracketing
            derivative_est = 1.0  # Default value if we can't estimate
            
            # Simple numerical differentiation to estimate derivative
            h = max(0.01 * y_guess, 0.001)
            error_plus = equation_func(x_values[i], y_guess + h)
            derivative_est = (error_plus - error) / h
            
            # Avoid division by very small numbers
            if abs(derivative_est) < 1e-10:
                derivative_est = np.sign(derivative_est) * 1e-10
                
            # Update estimate
            correction = error / derivative_est
            
            # Damping to prevent overshooting
            damping = min(1.0, 0.8 * abs(y_guess) / (abs(correction) + 1e-10))
            y_guess -= damping * correction
            
            # Ensure non-negative values for physical quantities like depth
            if y_guess < 0:
                y_guess = 0.001
        
        y_values[i] = y_guess
    
    # Solve in downstream direction
    for i in range(bc_idx + 1, n_points):
        # Initial guess: use previous value
        y_guess = y_values[i-1]
        
        # Iterative solution for current location (similar to upstream)
        for iter_count in range(max_iterations):
            # Calculate error from equation
            error = equation_func(x_values[i], y_guess)
            
            if abs(error) < tolerance:
                break
                
            # Similar numerical differentiation for downstream
            h = max(0.01 * y_guess, 0.001)
            error_plus = equation_func(x_values[i], y_guess + h)
            derivative_est = (error_plus - error) / h
            
            if abs(derivative_est) < 1e-10:
                derivative_est = np.sign(derivative_est) * 1e-10
                
            correction = error / derivative_est
            damping = min(1.0, 0.8 * abs(y_guess) / (abs(correction) + 1e-10))
            y_guess -= damping * correction
            
            if y_guess < 0:
                y_guess = 0.001
        
        y_values[i] = y_guess
    
    return {
        'x': x_values,
        'y': y_values
    }

def implicit_euler_method(
    equation_func: Callable[[float, float], float],
    y_start: float,
    x_range: Tuple[float, float],
    n_steps: int,
    tolerance: float = 1e-6,
    max_iterations: int = 20
) -> Dict[str, np.ndarray]:
    """
    Solve an ODE using the implicit Euler method.
    
    Parameters:
        equation_func (Callable): Function returning dy/dx at (x, y)
        y_start (float): Initial value of y
        x_range (Tuple[float, float]): (x_start, x_end)
        n_steps (int): Number of steps
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations per step
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    x_start, x_end = x_range
    dx = (x_end - x_start) / n_steps
    
    # Initialize arrays
    x_values = np.linspace(x_start, x_end, n_steps + 1)
    y_values = np.zeros_like(x_values)
    y_values[0] = y_start
    
    # Implicit Euler method
    for i in range(n_steps):
        # Initial guess for y[i+1]
        y_guess = y_values[i]
        
        # Newton iteration to solve implicit equation
        for j in range(max_iterations):
            # Equation to solve: y[i+1] - y[i] - dx * f(x[i+1], y[i+1]) = 0
            f = equation_func(x_values[i+1], y_guess)
            residual = y_guess - y_values[i] - dx * f
            
            if abs(residual) < tolerance:
                break
                
            # Approximate derivative of the residual
            h = max(0.001, 0.001 * abs(y_guess))
            f_perturbed = equation_func(x_values[i+1], y_guess + h)
            df_dy = (f_perturbed - f) / h
            dresidual_dy = 1.0 - dx * df_dy
            
            # Avoid division by very small numbers
            if abs(dresidual_dy) < 1e-10:
                dresidual_dy = np.sign(dresidual_dy) * 1e-10
                
            # Update guess
            correction = residual / dresidual_dy
            y_guess -= correction
        
        y_values[i+1] = y_guess
    
    return {
        'x': x_values,
        'y': y_values
    }

def crank_nicolson_method(
    equation_func: Callable[[float, float], float],
    y_start: float,
    x_range: Tuple[float, float],
    n_steps: int,
    tolerance: float = 1e-6,
    max_iterations: int = 20
) -> Dict[str, np.ndarray]:
    """
    Solve an ODE using the Crank-Nicolson method.
    
    Parameters:
        equation_func (Callable): Function returning dy/dx at (x, y)
        y_start (float): Initial value of y
        x_range (Tuple[float, float]): (x_start, x_end)
        n_steps (int): Number of steps
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations per step
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x and y arrays
    """
    x_start, x_end = x_range
    dx = (x_end - x_start) / n_steps
    
    # Initialize arrays
    x_values = np.linspace(x_start, x_end, n_steps + 1)
    y_values = np.zeros_like(x_values)
    y_values[0] = y_start
    
    # Crank-Nicolson method
    for i in range(n_steps):
        # Initial guess for y[i+1]
        y_guess = y_values[i]
        
        # Newton iteration to solve implicit equation
        for j in range(max_iterations):
            # Equation: y[i+1] - y[i] - 0.5 * dx * (f(x[i], y[i]) + f(x[i+1], y[i+1])) = 0
            f_i = equation_func(x_values[i], y_values[i])
            f_ip1 = equation_func(x_values[i+1], y_guess)
            residual = y_guess - y_values[i] - 0.5 * dx * (f_i + f_ip1)
            
            if abs(residual) < tolerance:
                break
                
            # Approximate derivative of the residual
            h = max(0.001, 0.001 * abs(y_guess))
            f_perturbed = equation_func(x_values[i+1], y_guess + h)
            df_dy = (f_perturbed - f_ip1) / h
            dresidual_dy = 1.0 - 0.5 * dx * df_dy
            
            # Avoid division by very small numbers
            if abs(dresidual_dy) < 1e-10:
                dresidual_dy = np.sign(dresidual_dy) * 1e-10
                
            # Update guess
            correction = residual / dresidual_dy
            y_guess -= correction
        
        y_values[i+1] = y_guess
    
    return {
        'x': x_values,
        'y': y_values
    }

def forward_time_centered_space(
    pde_func: Callable[[np.ndarray, float], np.ndarray],
    initial_condition: np.ndarray,
    x_values: np.ndarray,
    t_range: Tuple[float, float],
    n_steps: int
) -> Dict[str, np.ndarray]:
    """
    Solve a time-dependent PDE using the FTCS method.
    
    Parameters:
        pde_func (Callable): Function returning du/dt at each spatial point
        initial_condition (np.ndarray): Initial values of u at x_values
        x_values (np.ndarray): Spatial grid points
        t_range (Tuple[float, float]): (t_start, t_end)
        n_steps (int): Number of time steps
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x, t, and u arrays
    """
    t_start, t_end = t_range
    dt = (t_end - t_start) / n_steps
    
    # Initialize arrays
    t_values = np.linspace(t_start, t_end, n_steps + 1)
    n_x = len(x_values)
    u_values = np.zeros((n_steps + 1, n_x))
    
    # Set initial condition
    u_values[0, :] = initial_condition
    
    # FTCS method
    for i in range(n_steps):
        # Calculate du/dt at current time step
        du_dt = pde_func(u_values[i, :], t_values[i])
        
        # Update u using explicit Euler
        u_values[i+1, :] = u_values[i, :] + dt * du_dt
    
    return {
        'x': x_values,
        'time': t_values,
        'u': u_values
    }

def lax_wendroff_method(
    flux_func: Callable[[np.ndarray], np.ndarray],
    initial_condition: np.ndarray,
    x_values: np.ndarray,
    t_range: Tuple[float, float],
    n_steps: int
) -> Dict[str, np.ndarray]:
    """
    Solve a hyperbolic conservation law using the Lax-Wendroff method.
    
    Parameters:
        flux_func (Callable): Function returning flux F(u)
        initial_condition (np.ndarray): Initial values of u at x_values
        x_values (np.ndarray): Spatial grid points
        t_range (Tuple[float, float]): (t_start, t_end)
        n_steps (int): Number of time steps
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x, t, and u arrays
    """
    t_start, t_end = t_range
    dt = (t_end - t_start) / n_steps
    
    # Initialize arrays
    t_values = np.linspace(t_start, t_end, n_steps + 1)
    n_x = len(x_values)
    u_values = np.zeros((n_steps + 1, n_x))
    
    # Set initial condition
    u_values[0, :] = initial_condition
    
    # Calculate spatial step size
    dx = x_values[1] - x_values[0]
    
    # Lax-Wendroff method
    for i in range(n_steps):
        u_current = u_values[i, :]
        
        # Calculate flux at current time step
        flux = flux_func(u_current)
        
        # Calculate u at next time step
        for j in range(1, n_x - 1):
            # First term: u_j^n
            term1 = u_current[j]
            
            # Second term: -0.5 * dt/dx * (F_{j+1} - F_{j-1})
            term2 = -0.5 * dt / dx * (flux[j+1] - flux[j-1])
            
            # Third term: 0.5 * (dt/dx)^2 * (F_{j+1} - 2*F_j + F_{j-1})
            term3 = 0.5 * (dt / dx)**2 * (flux[j+1] - 2 * flux[j] + flux[j-1])
            
            u_values[i+1, j] = term1 + term2 + term3
        
        # Apply boundary conditions
        u_values[i+1, 0] = u_current[0]  # Simple Dirichlet at left boundary
        u_values[i+1, -1] = u_current[-1]  # Simple Dirichlet at right boundary
    
    return {
        'x': x_values,
        'time': t_values,
        'u': u_values
    }

def preissmann_scheme(
    equations_func: Callable[[np.ndarray, np.ndarray, float, float], Tuple[np.ndarray, np.ndarray]],
    initial_values: np.ndarray,
    boundary_conditions: Callable[[float], Tuple[float, Optional[float]]],
    x_values: np.ndarray,
    t_range: Tuple[float, float],
    n_steps: int,
    theta: float = 0.6,
    max_iterations: int = 20,
    tolerance: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Solve Saint-Venant equations using the Preissmann scheme.
    
    Parameters:
        equations_func (Callable): Function returning coefficients for the equations
        initial_values (np.ndarray): Initial values [h, Q] at each x location
        boundary_conditions (Callable): Function returning upstream and downstream values at time t
        x_values (np.ndarray): Spatial grid points
        t_range (Tuple[float, float]): (t_start, t_end)
        n_steps (int): Number of time steps
        theta (float): Weighting factor for time discretization (0.5-1.0)
        max_iterations (int): Maximum iterations per step
        tolerance (float): Convergence tolerance
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with x, t, h, and Q arrays
    """
    t_start, t_end = t_range
    dt = (t_end - t_start) / n_steps
    
    # Initialize arrays
    t_values = np.linspace(t_start, t_end, n_steps + 1)
    n_x = len(x_values)
    h_values = np.zeros((n_steps + 1, n_x))
    Q_values = np.zeros((n_steps + 1, n_x))
    
    # Set initial condition
    h_values[0, :] = initial_values[:, 0]
    Q_values[0, :] = initial_values[:, 1]
    
    # Calculate spatial step size
    dx = x_values[1] - x_values[0]
    
    # Time stepping
    for i in range(n_steps):
        # Get current time and next time
        t_current = t_values[i]
        t_next = t_values[i+1]
        
        # Get boundary conditions
        upstream_bc, downstream_bc = boundary_conditions(t_next)
        
        # Initialize guess for next time step
        h_guess = h_values[i, :].copy()
        Q_guess = Q_values[i, :].copy()
        
        # Apply boundary conditions to guess
        if upstream_bc is not None:
            Q_guess[0] = upstream_bc
        if downstream_bc is not None:
            h_guess[-1] = downstream_bc
        
        # Newton iteration to solve implicit system
        for j in range(max_iterations):
            # Calculate coefficients and residuals
            coeffs, residuals = equations_func(
                np.vstack((h_values[i, :], Q_values[i, :])),
                np.vstack((h_guess, Q_guess)),
                dx, dt
            )
            
            # Check convergence
            if np.max(np.abs(residuals)) < tolerance:
                break
            
            # Solve the linear system
            # This is a simplified approach - full implementation would use a banded matrix solver
            # Here we use a simple direct solve for demonstration
            corrections = np.linalg.solve(coeffs, -residuals)
            
            # Apply corrections
            for k in range(n_x):
                h_guess[k] += corrections[2*k]
                Q_guess[k] += corrections[2*k+1]
            
            # Re-apply boundary conditions
            if upstream_bc is not None:
                Q_guess[0] = upstream_bc
            if downstream_bc is not None:
                h_guess[-1] = downstream_bc
        
        # Store results
        h_values[i+1, :] = h_guess
        Q_values[i+1, :] = Q_guess
    
    return {
        'x': x_values,
        'time': t_values,
        'h': h_values,
        'Q': Q_values
    }