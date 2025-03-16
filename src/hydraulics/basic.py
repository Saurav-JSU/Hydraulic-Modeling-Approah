"""
Basic hydraulic functions and constants for open channel flow analysis.

This module contains fundamental hydraulic calculations and constants used
throughout the dam and channel flow analysis project.
"""

import math
from typing import Dict, Optional, Union

# Physical constants
GRAVITY = 9.81  # Gravitational acceleration (m/s²)
WATER_DENSITY = 1000  # Water density (kg/m³)

def hydraulic_radius(area: float, wetted_perimeter: float) -> float:
    """
    Calculate hydraulic radius from area and wetted perimeter.
    
    Parameters:
        area (float): Cross-sectional area of flow (m²)
        wetted_perimeter (float): Wetted perimeter (m)
        
    Returns:
        float: Hydraulic radius (m)
        
    Raises:
        ValueError: If wetted_perimeter is zero or negative
    """
    if wetted_perimeter <= 0:
        raise ValueError("Wetted perimeter must be positive")
    
    return area / wetted_perimeter

def section_properties(depth: float, bottom_width: float, side_slope: float = 0) -> Dict[str, float]:
    """
    Calculate cross-sectional properties for trapezoidal channel.
    For rectangular channel, set side_slope = 0.
    
    Parameters:
        depth (float): Water depth (m)
        bottom_width (float): Bottom width of channel (m)
        side_slope (float, optional): Side slope (horizontal/vertical). Defaults to 0 (rectangular).
        
    Returns:
        Dict[str, float]: Dictionary containing:
            - area: Cross-sectional area (m²)
            - wetted_perimeter: Wetted perimeter (m)
            - top_width: Width of water surface (m)
            - hydraulic_radius: Hydraulic radius (m)
            
    Raises:
        ValueError: If depth or bottom_width is negative
    """
    if depth < 0:
        raise ValueError("Depth must be non-negative")
    if bottom_width < 0:
        raise ValueError("Bottom width must be non-negative")
    
    # For zero depth, return zero values
    if depth == 0:
        return {
            "area": 0,
            "wetted_perimeter": bottom_width,  # Corrected: wetted perimeter equals bottom width when depth is zero
            "top_width": bottom_width,
            "hydraulic_radius": 0
        }
    
    # Calculate top width
    top_width = bottom_width + 2 * side_slope * depth
    
    # Calculate area
    area = (bottom_width + top_width) * depth / 2
    
    # Calculate wetted perimeter
    if side_slope == 0:  # Rectangular
        wetted_perimeter = bottom_width + 2 * depth
    else:  # Trapezoidal
        sloped_side_length = depth * math.sqrt(1 + side_slope**2)
        wetted_perimeter = bottom_width + 2 * sloped_side_length
    
    # Calculate hydraulic radius
    hyd_radius = area / wetted_perimeter
    
    return {
        "area": area,
        "wetted_perimeter": wetted_perimeter,
        "top_width": top_width,
        "hydraulic_radius": hyd_radius
    }

def critical_depth(discharge: float, top_width: float) -> float:
    """
    Calculate critical depth for rectangular channels.
    
    Parameters:
        discharge (float): Flow rate (m³/s)
        top_width (float): Width of water surface (m)
        
    Returns:
        float: Critical depth (m)
        
    Raises:
        ValueError: If discharge or top_width is negative
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if top_width <= 0:
        raise ValueError("Top width must be positive")
    
    # For zero discharge, critical depth is zero
    if discharge == 0:
        return 0
    
    # Calculate critical depth using the formula: y_c = (q²/g)^(1/3)
    # where q = discharge per unit width
    unit_discharge = discharge / top_width
    critical_depth = (unit_discharge**2 / GRAVITY)**(1/3)
    
    return critical_depth

def critical_depth_general(discharge: float, section_props_func, *args, **kwargs) -> float:
    """
    Calculate critical depth for any channel shape using an iterative approach.
    
    Parameters:
        discharge (float): Flow rate (m³/s)
        section_props_func: Function that returns section properties given a depth
        *args, **kwargs: Additional arguments to pass to section_props_func
        
    Returns:
        float: Critical depth (m)
        
    Raises:
        ValueError: If discharge is negative
        RuntimeError: If iteration fails to converge
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    
    # For zero discharge, critical depth is zero
    if discharge == 0:
        return 0
    
    # Initial guess for critical depth
    # For a rectangular channel, try the simple formula first
    props = section_props_func(1.0, *args, **kwargs)
    if 'top_width' in props:
        y_c = critical_depth(discharge, props['top_width'])
    else:
        y_c = 1.0  # Default initial guess
    
    # Iterative solution using Newton-Raphson method
    tolerance = 1e-6
    max_iterations = 50
    iterations = 0
    
    while iterations < max_iterations:
        # Get section properties at current depth
        props = section_props_func(y_c, *args, **kwargs)
        
        # Calculate specific energy at critical condition
        area = props['area']
        top_width = props['top_width']
        
        # At critical flow, specific energy is minimized where:
        # Q^2*T/gA^3 = 1, where T is top width and A is area
        critical_param = discharge**2 * top_width / (GRAVITY * area**3)
        
        # Calculate error
        error = critical_param - 1.0
        
        # Check convergence
        if abs(error) < tolerance:
            return y_c
        
        # Calculate derivative of error function
        delta = 0.001 * y_c  # Small depth change for numerical derivative
        props_delta = section_props_func(y_c + delta, *args, **kwargs)
        area_delta = props_delta['area']
        top_width_delta = props_delta['top_width']
        
        critical_param_delta = discharge**2 * top_width_delta / (GRAVITY * area_delta**3)
        derivative = (critical_param_delta - critical_param) / delta
        
        # Update depth using Newton-Raphson
        if abs(derivative) < 1e-10:  # Prevent division by near-zero
            y_c = y_c * 1.1  # Simple adjustment if derivative is too small
        else:
            y_c = y_c - error / derivative
        
        # Ensure depth remains positive
        y_c = max(y_c, 1e-6)
        
        iterations += 1
    
    raise RuntimeError("Critical depth calculation failed to converge")

def froude_number(velocity: float, depth: float) -> float:
    """
    Calculate Froude number.
    
    Parameters:
        velocity (float): Average flow velocity (m/s)
        depth (float): Water depth (m)
        
    Returns:
        float: Froude number (dimensionless)
        
    Raises:
        ValueError: If depth is negative
    """
    if depth <= 0:
        raise ValueError("Depth must be positive for Froude number calculation")
    
    return velocity / math.sqrt(GRAVITY * depth)

def froude_number_general(velocity: float, area: float, top_width: float) -> float:
    """
    Calculate Froude number for any channel shape.
    
    Parameters:
        velocity (float): Average flow velocity (m/s)
        area (float): Cross-sectional area of flow (m²)
        top_width (float): Width of water surface (m)
        
    Returns:
        float: Froude number (dimensionless)
        
    Raises:
        ValueError: If invalid inputs are provided
    """
    if area <= 0:
        raise ValueError("Area must be positive for Froude number calculation")
    if top_width <= 0:
        raise ValueError("Top width must be positive for Froude number calculation")
    
    # Calculate hydraulic depth (area/top_width)
    hydraulic_depth = area / top_width
    
    # Calculate Froude number using hydraulic depth
    return velocity / math.sqrt(GRAVITY * hydraulic_depth)

def flow_classification(froude_number: float) -> str:
    """
    Classify flow based on Froude number.
    
    Parameters:
        froude_number (float): Froude number
        
    Returns:
        str: Flow classification ('Subcritical', 'Critical', or 'Supercritical')
    """
    if froude_number < 0.99:
        return "Subcritical"
    elif froude_number > 1.01:
        return "Supercritical"
    else:
        return "Critical"

def reynolds_number(velocity: float, hydraulic_radius: float, 
                   kinematic_viscosity: float = 1.0e-6) -> float:
    """
    Calculate Reynolds number.
    
    Parameters:
        velocity (float): Average flow velocity (m/s)
        hydraulic_radius (float): Hydraulic radius (m)
        kinematic_viscosity (float, optional): Kinematic viscosity (m²/s).
            Defaults to 1.0e-6 (for water at 20°C).
        
    Returns:
        float: Reynolds number (dimensionless)
        
    Raises:
        ValueError: If hydraulic_radius is negative
    """
    if hydraulic_radius <= 0:
        raise ValueError("Hydraulic radius must be positive for Reynolds number calculation")
    
    # Use factor of 4 to convert from hydraulic radius to hydraulic diameter
    # This is standard practice in open channel hydraulics
    return 4 * velocity * hydraulic_radius / kinematic_viscosity