"""
Manning's equation implementations for open channel flow analysis.

This module contains functions for various applications of Manning's equation 
in open channel hydraulics.
"""

import math
from typing import Dict, Optional, List
from . import basic

def manning_velocity(hydraulic_radius: float, slope: float, roughness: float) -> float:
    """
    Calculate velocity using Manning's equation.
    
    Parameters:
        hydraulic_radius (float): Hydraulic radius (m)
        slope (float): Channel bed slope (m/m)
        roughness (float): Manning's roughness coefficient (n)
        
    Returns:
        float: Average flow velocity (m/s)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if hydraulic_radius <= 0:
        raise ValueError("Hydraulic radius must be positive")
    if slope <= 0:
        raise ValueError("Slope must be positive")
    if roughness <= 0:
        raise ValueError("Roughness coefficient must be positive")
    
    # Manning's equation: V = (1/n) * R^(2/3) * S^(1/2)
    velocity = (1.0 / roughness) * hydraulic_radius**(2/3) * math.sqrt(slope)
    
    return velocity

def discharge(depth: float, bottom_width: float, slope: float, roughness: float, 
             side_slope: float = 0) -> float:
    """
    Calculate discharge for given depth in channel using Manning's equation.
    
    Parameters:
        depth (float): Water depth (m)
        bottom_width (float): Bottom width of channel (m)
        slope (float): Channel bed slope (m/m)
        roughness (float): Manning's roughness coefficient (n)
        side_slope (float, optional): Side slope (horizontal/vertical). 
            Defaults to 0 (rectangular).
        
    Returns:
        float: Discharge (m³/s)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth < 0:
        raise ValueError("Depth must be non-negative")
    if bottom_width < 0:
        raise ValueError("Bottom width must be non-negative")
    if slope <= 0:
        raise ValueError("Slope must be positive")
    if roughness <= 0:
        raise ValueError("Roughness coefficient must be positive")
    
    # For zero depth, discharge is zero
    if depth == 0:
        return 0
    
    # Calculate section properties
    props = basic.section_properties(depth, bottom_width, side_slope)
    area = props["area"]
    hydraulic_radius = props["hydraulic_radius"]
    
    # Calculate velocity using Manning's equation
    velocity = manning_velocity(hydraulic_radius, slope, roughness)
    
    # Discharge = Area * Velocity
    return area * velocity

def normal_depth(target_discharge: float, bottom_width: float, slope: float, roughness: float, 
                side_slope: float = 0, tolerance: float = 0.0001, max_iterations: int = 100) -> float:
    """
    Calculate normal depth for given discharge in channel using Manning's equation.
    Uses numerical method (secant method) to solve.
    
    Parameters:
        target_discharge (float): Flow rate (m³/s)
        bottom_width (float): Bottom width of channel (m)
        slope (float): Channel bed slope (m/m)
        roughness (float): Manning's roughness coefficient (n)
        side_slope (float, optional): Side slope (horizontal/vertical). 
            Defaults to 0 (rectangular).
        tolerance (float, optional): Error tolerance for iteration. Defaults to 0.0001.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
        
    Returns:
        float: Normal depth (m)
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If solution doesn't converge within max_iterations
    """
    if target_discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if bottom_width <= 0:
        raise ValueError("Bottom width must be positive")
    if slope <= 0:
        raise ValueError("Slope must be positive")
    if roughness <= 0:
        raise ValueError("Roughness coefficient must be positive")
    
    # For zero discharge, normal depth is zero
    if target_discharge == 0:
        return 0
    
    # Function to find root of: f(y) = Q_manning(y) - Q_target
    def depth_function(y: float) -> float:
        q_manning = discharge(y, bottom_width, slope, roughness, side_slope)
        return q_manning - target_discharge
    
    # Initial guesses for depth
    # Use approximation for rectangular channel as starting point
    y1 = (target_discharge * roughness / (bottom_width * math.sqrt(slope)))**(3/5)
    
    # Second guess is slightly larger
    y2 = 1.1 * y1
    
    # Evaluate function at initial guesses
    f1 = depth_function(y1)
    
    # Secant method iteration
    for i in range(max_iterations):
        f2 = depth_function(y2)
        
        # Check if we've converged
        if abs(f2) < tolerance:
            return y2
        
        # Calculate next guess using secant method
        if f2 - f1 == 0:  # Avoid division by zero
            y_new = (y1 + y2) / 2  # Use bisection instead
        else:
            y_new = y2 - f2 * (y2 - y1) / (f2 - f1)
        
        # Ensure depth stays positive
        if y_new <= 0:
            y_new = (y1 + y2) / 2
        
        # Update values for next iteration
        y1, f1 = y2, f2
        y2 = y_new
        
        # Check if converged based on depth change
        if abs(y2 - y1) < tolerance:
            return y2
    
    # If we reached max iterations without converging
    raise RuntimeError(f"Normal depth calculation did not converge within {max_iterations} iterations")

def shear_stress(hydraulic_radius: float, slope: float, specific_weight: float = 9810) -> float:
    """
    Calculate boundary shear stress.
    
    Parameters:
        hydraulic_radius (float): Hydraulic radius (m)
        slope (float): Channel bed slope (m/m)
        specific_weight (float, optional): Specific weight of water (N/m³). 
            Defaults to 9810 N/m³.
        
    Returns:
        float: Boundary shear stress (N/m²)
        
    Raises:
        ValueError: If hydraulic_radius or slope is negative
    """
    if hydraulic_radius < 0:
        raise ValueError("Hydraulic radius must be non-negative")
    if slope < 0:
        raise ValueError("Slope must be non-negative")
    
    # Shear stress = γ * R * S
    return specific_weight * hydraulic_radius * slope

def velocity_distribution(y_values: List[float], total_depth: float, average_velocity: float,
                         distribution_type: str = 'log') -> List[float]:
    """
    Calculate velocity distribution in a channel.
    
    Parameters:
        y_values (List[float]): List of heights from bed where velocities are to be calculated (m)
        total_depth (float): Total water depth (m)
        average_velocity (float): Cross-sectional average velocity (m/s)
        distribution_type (str, optional): Type of distribution ('log' or 'power'). 
            Defaults to 'log'.
        
    Returns:
        List[float]: List of velocities corresponding to each height in y_values
        
    Raises:
        ValueError: If inputs are invalid or distribution_type is unknown
    """
    if total_depth <= 0:
        raise ValueError("Total depth must be positive")
    if average_velocity < 0:
        raise ValueError("Average velocity must be non-negative")
    
    velocities = []
    
    if distribution_type.lower() == 'log':
        # Improved logarithmic velocity distribution based on turbulent flow principles
        # Using the universal velocity profile for turbulent open channel flow
        
        kappa = 0.41  # von Karman's constant
        
        # Estimate bed roughness height (y0) based on typical relative roughness
        # This approximation reflects moderate channel bed conditions
        bed_roughness = 0.03 * total_depth / 30
        
        # Calculate shear velocity (u*) from average velocity using the
        # theoretical relationship for a logarithmic profile:
        # u_avg / u* ≈ (1/κ) * [ln(h/y0) - 1]
        u_star = average_velocity * kappa / (math.log(total_depth / bed_roughness) - 1)
        
        for y in y_values:
            if y <= 0 or y > total_depth:
                raise ValueError(f"Height value {y} is outside valid range (0, {total_depth}]")
            
            # Prevent mathematical singularity near the bed by setting a minimum height
            effective_y = max(y, bed_roughness)
            
            # Calculate velocity using the law of the wall for turbulent flow
            # u(y) = (u* / κ) * ln(y / y0)
            u = (u_star / kappa) * math.log(effective_y / bed_roughness)
            
            # Ensure velocity is non-negative (physical constraint)
            velocities.append(max(0, u))
    
    elif distribution_type.lower() == 'power':
        # Power law velocity distribution
        # u(y) = u_max * (y/h)^(1/m)
        
        # Use m = 7 for fully developed turbulent flow in open channels
        # This is more appropriate than the previous value of 6
        m = 7
        
        # Calculate maximum velocity from average velocity using the theoretical
        # relationship for power law: u_max / u_avg = (m+1)/m
        u_max = average_velocity * (m + 1) / m
        
        for y in y_values:
            if y <= 0 or y > total_depth:
                raise ValueError(f"Height value {y} is outside valid range (0, {total_depth}]")
            
            rel_depth = y / total_depth
            # Calculate velocity using power law with the proper exponent
            u = u_max * rel_depth**(1/m)
            velocities.append(u)
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}. Use 'log' or 'power'.")
    
    return velocities