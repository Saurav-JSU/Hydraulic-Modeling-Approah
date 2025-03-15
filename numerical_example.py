"""
Example script demonstrating the use of the numerical module.

This script shows how to solve hydraulic differential equations
using different numerical methods.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
# This is for demonstration only - in a real project, you'd install the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.numerical import (
    direct_step_method,
    standard_step_method,
    runge_kutta_4,
    adaptive_rk4,
    implicit_euler_method
)

def solve_gradually_varied_flow():
    """Solve a gradually varied flow problem using different methods."""
    print("\nGradually Varied Flow Solution")
    print("-" * 50)
    
    # Define the GVF equation (dy/dx)
    # For a simple rectangular channel with width=10m, Manning's n=0.015, S0=0.001, Q=20m³/s
    
    def gvf_equation(y):
        """Right-hand side of the GVF equation dy/dx = f(y)."""
        # Channel parameters
        width = 10.0  # m
        roughness = 0.015  # Manning's n
        slope = 0.001  # m/m
        discharge = 20.0  # m³/s
        g = 9.81  # Gravitational acceleration (m/s²)
        
        # Calculate area and hydraulic radius
        area = width * y
        wetted_perimeter = width + 2 * y
        hydraulic_radius = area / wetted_perimeter
        
        # Calculate velocity and Froude number
        velocity = discharge / area
        froude = velocity / math.sqrt(g * y)
        
        # Calculate friction slope using Manning's equation
        friction_slope = (roughness * velocity)**2 / hydraulic_radius**(4/3)
        
        # GVF equation: dy/dx = (S0 - Sf) / (1 - Fr²)
        numerator = slope - friction_slope
        denominator = 1 - froude**2
        
        return numerator / denominator
    
    # Initial and target depths
    normal_depth = 1.41  # m (calculated separately)
    critical_depth = 0.97  # m (calculated separately)
    
    # Solve using direct step method
    # M2 profile: from normal depth to near critical depth
    print("\nSolving M2 profile using direct step method...")
    results_direct = direct_step_method(gvf_equation, normal_depth, 1.1*critical_depth)
    
    # Solve the same problem using RK4
    print("\nSolving the same problem using RK4...")
    
    def gvf_equation_rk4(x, y):
        """GVF equation in the form required by RK4: dy/dx = f(x, y)."""
        return gvf_equation(y)
    
    max_distance = np.max(results_direct['x'])
    results_rk4 = runge_kutta_4(gvf_equation_rk4, normal_depth, (0, max_distance), 100)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    plt.plot(results_direct['x'], results_direct['y'], 'b-', label='Direct Step Method')
    plt.plot(results_rk4['x'], results_rk4['y'], 'r--', label='Runge-Kutta 4')
    
    plt.axhline(y=normal_depth, color='g', linestyle=':', label=f'Normal Depth ({normal_depth:.3f} m)')
    plt.axhline(y=critical_depth, color='m', linestyle=':', label=f'Critical Depth ({critical_depth:.3f} m)')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Water Depth (m)')
    plt.title('Gradually Varied Flow - M2 Profile')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('gvf_numerical_comparison.png')
    plt.close()
    
    print("\nGVF solution comparison saved as 'gvf_numerical_comparison.png'")
    
    # Compare solution accuracy
    # Interpolate direct step solution at RK4 x values
    y_direct_interp = np.interp(results_rk4['x'], results_direct['x'], results_direct['y'])
    
    # Calculate maximum absolute difference
    max_diff = np.max(np.abs(y_direct_interp - results_rk4['y']))
    
    print(f"Maximum difference between methods: {max_diff:.6f} m")

def compare_numerical_schemes():
    """Compare different numerical schemes for solving a test ODE."""
    print("\nNumerical Scheme Comparison")
    print("-" * 50)
    
    # Define a test ODE: dy/dx = -2y
    # Analytical solution: y = y0 * exp(-2x)
    
    def test_equation(x, y):
        """Test equation: dy/dx = -2y."""
        return -2 * y
    
    # Initial condition
    y0 = 1.0
    
    # Analytical solution
    def analytical_solution(x):
        """Analytical solution: y = y0 * exp(-2x)."""
        return y0 * np.exp(-2 * x)
    
    # Solve using different methods
    x_range = (0, 2)
    print("\nSolving test ODE with different numerical schemes...")
    
    # RK4 with different step sizes
    results_rk4_10 = runge_kutta_4(test_equation, y0, x_range, 10)
    results_rk4_50 = runge_kutta_4(test_equation, y0, x_range, 50)
    
    # Adaptive RK4
    results_adaptive = adaptive_rk4(test_equation, y0, x_range)
    
    # Implicit Euler
    results_implicit = implicit_euler_method(test_equation, y0, x_range, 50)
    
    # Calculate analytical solution
    x_fine = np.linspace(x_range[0], x_range[1], 200)
    y_analytical = analytical_solution(x_fine)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(x_fine, y_analytical, 'k-', label='Analytical Solution')
    plt.plot(results_rk4_10['x'], results_rk4_10['y'], 'b-', marker='o', markersize=4, label='RK4 (10 steps)')
    plt.plot(results_rk4_50['x'], results_rk4_50['y'], 'g-', marker='.', markersize=4, label='RK4 (50 steps)')
    plt.plot(results_adaptive['x'], results_adaptive['y'], 'r-', marker='x', markersize=4, label=f'Adaptive RK4 ({len(results_adaptive["x"])} steps)')
    plt.plot(results_implicit['x'], results_implicit['y'], 'm-', marker='+', markersize=4, label='Implicit Euler (50 steps)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Numerical Schemes')
    plt.grid(True)
    plt.legend()
    
    # Calculate and plot errors
    plt.subplot(2, 1, 2)
    
    # Analytical solutions at each scheme's x values
    y_exact_rk4_10 = analytical_solution(results_rk4_10['x'])
    y_exact_rk4_50 = analytical_solution(results_rk4_50['x'])
    y_exact_adaptive = analytical_solution(results_adaptive['x'])
    y_exact_implicit = analytical_solution(results_implicit['x'])
    
    # Calculate errors
    error_rk4_10 = np.abs(results_rk4_10['y'] - y_exact_rk4_10)
    error_rk4_50 = np.abs(results_rk4_50['y'] - y_exact_rk4_50)
    error_adaptive = np.abs(results_adaptive['y'] - y_exact_adaptive)
    error_implicit = np.abs(results_implicit['y'] - y_exact_implicit)
    
    # Plot errors
    plt.semilogy(results_rk4_10['x'], error_rk4_10, 'b-', marker='o', markersize=4, label='RK4 (10 steps)')
    plt.semilogy(results_rk4_50['x'], error_rk4_50, 'g-', marker='.', markersize=4, label='RK4 (50 steps)')
    plt.semilogy(results_adaptive['x'], error_adaptive, 'r-', marker='x', markersize=4, label='Adaptive RK4')
    plt.semilogy(results_implicit['x'], error_implicit, 'm-', marker='+', markersize=4, label='Implicit Euler')
    
    plt.xlabel('x')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('numerical_scheme_comparison.png')
    plt.close()
    
    print("\nNumerical scheme comparison saved as 'numerical_scheme_comparison.png'")
    
    # Print error statistics
    print("\nError Statistics:")
    print(f"RK4 (10 steps): Max error = {np.max(error_rk4_10):.2e}, Mean error = {np.mean(error_rk4_10):.2e}")
    print(f"RK4 (50 steps): Max error = {np.max(error_rk4_50):.2e}, Mean error = {np.mean(error_rk4_50):.2e}")
    print(f"Adaptive RK4: Max error = {np.max(error_adaptive):.2e}, Mean error = {np.mean(error_adaptive):.2e}")
    print(f"Implicit Euler: Max error = {np.max(error_implicit):.2e}, Mean error = {np.mean(error_implicit):.2e}")

def solve_boundary_value_problem():
    """Solve a hydraulic boundary value problem using the standard step method."""
    print("\nBoundary Value Problem Solution")
    print("-" * 50)
    
    # Define the standard step equation for a simple channel with a bump
    
    def error_equation(x, y_guess):
        """
        Error function for the standard step method.
        Returns the error in energy balance at the given location with the guessed depth.
        """
        # Channel parameters
        width = 10.0  # m
        roughness = 0.015  # Manning's n
        discharge = 20.0  # m³/s
        g = 9.81  # Gravitational acceleration (m/s²)
        
        # Channel bed elevation (with a bump)
        def bed_elevation(x_val):
            # Parabolic bump centered at x=50m
            if 25 <= x_val <= 75:
                return 0.5 * (1 - ((x_val - 50) / 25)**2)
            else:
                return 0.0
        
        # Energy calculations
        # Area and velocity at current guess
        area_guess = width * y_guess
        velocity_guess = discharge / area_guess
        specific_energy_guess = y_guess + velocity_guess**2 / (2 * g)
        
        # Bed elevation and energy at current location
        z_current = bed_elevation(x)
        energy_current = z_current + specific_energy_guess
        
        # Target energy level (assuming constant energy throughout the channel)
        target_energy = 2.0  # m (example value)
        
        # Return the error in energy
        return energy_current - target_energy
    
    # Define x stations
    x_stations = np.linspace(0, 100, 101)
    
    # Known water depth at the downstream boundary
    downstream_depth = 1.5  # m
    boundary_condition = (100.0, downstream_depth)
    
    # Solve using standard step method
    print("\nSolving boundary value problem using standard step method...")
    results = standard_step_method(x_stations, error_equation, boundary_condition)
    
    # Calculate bed elevation for plotting
    def bed_elevation(x_val):
        # Parabolic bump centered at x=50m
        if 25 <= x_val <= 75:
            return 0.5 * (1 - ((x_val - 50) / 25)**2)
        else:
            return 0.0
    
    bed_profile = np.array([bed_elevation(x) for x in x_stations])
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.plot(x_stations, bed_profile, 'k-', label='Channel Bed')
    plt.plot(x_stations, bed_profile + results['y'], 'b-', label='Water Surface')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Water Surface Profile over a Bump')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('boundary_value_problem.png')
    plt.close()
    
    print("\nBoundary value problem solution saved as 'boundary_value_problem.png'")

if __name__ == "__main__":
    # Run example analyses
    solve_gradually_varied_flow()
    compare_numerical_schemes()
    solve_boundary_value_problem()
    
    print("\nExamples completed successfully.")