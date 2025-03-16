"""
Energy and momentum principles for open channel flow analysis.

This module contains functions for analyzing energy and momentum in open channels,
including specific energy, momentum functions, and hydraulic jump calculations.
"""

import math
from typing import Dict, Optional, Tuple, Callable
from . import basic

def specific_energy(depth: float, velocity: float) -> float:
    """
    Calculate specific energy at a channel section.
    
    Parameters:
        depth (float): Water depth (m)
        velocity (float): Average flow velocity (m/s)
        
    Returns:
        float: Specific energy (m)
        
    Raises:
        ValueError: If depth is negative
    """
    if depth < 0:
        raise ValueError("Depth must be non-negative")
    
    # Specific energy = depth + velocity head
    # E = y + v²/(2g)
    velocity_head = velocity**2 / (2 * basic.GRAVITY)
    return depth + velocity_head

def specific_energy_from_discharge(depth: float, discharge: float, width: float) -> float:
    """
    Calculate specific energy from discharge for a rectangular channel.
    
    Parameters:
        depth (float): Water depth (m)
        discharge (float): Flow rate (m³/s)
        width (float): Channel width (m)
        
    Returns:
        float: Specific energy (m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth <= 0:
        raise ValueError("Depth must be positive")
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if width <= 0:
        raise ValueError("Width must be positive")
    
    # Handle zero discharge case
    if discharge == 0:
        return depth  # No velocity head when discharge is zero
    
    # Calculate velocity
    area = depth * width
    velocity = discharge / area
    
    # Calculate specific energy
    return specific_energy(depth, velocity)

def specific_energy_general(depth: float, discharge: float, 
                           section_props_func: Callable, *args, **kwargs) -> float:
    """
    Calculate specific energy from discharge for any channel shape.
    
    Parameters:
        depth (float): Water depth (m)
        discharge (float): Flow rate (m³/s)
        section_props_func: Function that returns section properties given a depth
        *args, **kwargs: Additional arguments to pass to section_props_func
        
    Returns:
        float: Specific energy (m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth <= 0:
        raise ValueError("Depth must be positive")
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    
    # Handle zero discharge case
    if discharge == 0:
        return depth  # No velocity head when discharge is zero
    
    # Get section properties
    props = section_props_func(depth, *args, **kwargs)
    
    # Check if area is available
    if 'area' not in props:
        raise ValueError("Section properties function must return 'area'")
    
    area = props['area']
    
    if area <= 0:
        raise ValueError("Cross-sectional area must be positive")
    
    # Calculate velocity
    velocity = discharge / area
    
    # Calculate specific energy
    return specific_energy(depth, velocity)

def critical_specific_energy(discharge: float, width: float) -> float:
    """
    Calculate minimum specific energy (at critical depth) for a rectangular channel.
    
    Parameters:
        discharge (float): Flow rate (m³/s)
        width (float): Channel width (m)
        
    Returns:
        float: Critical specific energy (m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if width <= 0:
        raise ValueError("Width must be positive")
    
    # Handle zero discharge case
    if discharge == 0:
        return 0
    
    # Calculate critical depth
    yc = basic.critical_depth(discharge, width)
    
    # At critical depth for rectangular channels, E = 3/2 * yc
    # This is derived from the fact that velocity head = yc/2 at critical flow
    return 1.5 * yc

def critical_specific_energy_general(discharge: float, 
                                    section_props_func: Callable, 
                                    *args, **kwargs) -> float:
    """
    Calculate minimum specific energy (at critical depth) for any channel shape.
    
    Parameters:
        discharge (float): Flow rate (m³/s)
        section_props_func: Function that returns section properties given a depth
        *args, **kwargs: Additional arguments to pass to section_props_func
        
    Returns:
        float: Critical specific energy (m)
        
    Raises:
        ValueError: If discharge is negative
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    
    # Handle zero discharge case
    if discharge == 0:
        return 0
    
    # Find critical depth iteratively
    try:
        # Use the general critical depth function from basic module if it exists
        if hasattr(basic, 'critical_depth_general'):
            yc = basic.critical_depth_general(discharge, section_props_func, *args, **kwargs)
        else:
            # Fallback to simplified calculation for rectangular channels
            # Get section properties to estimate width
            props = section_props_func(1.0, *args, **kwargs)
            if 'top_width' in props:
                yc = basic.critical_depth(discharge, props['top_width'])
            else:
                raise ValueError("Section properties must include 'top_width'")
    except Exception as e:
        raise ValueError(f"Failed to calculate critical depth: {str(e)}")
    
    # Calculate specific energy at critical depth
    props = section_props_func(yc, *args, **kwargs)
    area = props['area']
    
    # Calculate velocity at critical depth
    velocity = discharge / area
    
    # Calculate specific energy
    return specific_energy(yc, velocity)

def energy_grade_line(water_surface_elevation: float, velocity: float) -> float:
    """
    Calculate energy grade line elevation.
    
    Parameters:
        water_surface_elevation (float): Water surface elevation (m)
        velocity (float): Average flow velocity (m/s)
        
    Returns:
        float: Energy grade line elevation (m)
    """
    # Energy grade line = water surface elevation + velocity head
    velocity_head = velocity**2 / (2 * basic.GRAVITY)
    return water_surface_elevation + velocity_head

def momentum_function(depth: float, discharge: float, width: float) -> float:
    """
    Calculate specific momentum function for a rectangular channel.
    
    Parameters:
        depth (float): Water depth (m)
        discharge (float): Flow rate (m³/s)
        width (float): Channel width (m)
        
    Returns:
        float: Momentum function value (m²)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth <= 0:
        raise ValueError("Depth must be positive")
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if width <= 0:
        raise ValueError("Width must be positive")
    
    # Handle zero discharge case
    if discharge == 0:
        return depth**2 / 2  # Only hydrostatic component remains
    
    # Calculate momentum function
    # M = y²/2 + q²/(gy)
    # where q = Q/W (discharge per unit width)
    q = discharge / width
    return (depth**2 / 2) + (q**2 / (basic.GRAVITY * depth))

def momentum_function_general(depth: float, discharge: float, 
                             section_props_func: Callable, *args, **kwargs) -> float:
    """
    Calculate specific momentum function for any channel shape.
    
    Parameters:
        depth (float): Water depth (m)
        discharge (float): Flow rate (m³/s)
        section_props_func: Function that returns section properties given a depth
        *args, **kwargs: Additional arguments to pass to section_props_func
        
    Returns:
        float: Momentum function value (m²)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth <= 0:
        raise ValueError("Depth must be positive")
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    
    # Get section properties
    props = section_props_func(depth, *args, **kwargs)
    
    # Check required properties
    required_props = ['area', 'top_width']
    for prop in required_props:
        if prop not in props:
            raise ValueError(f"Section properties function must return '{prop}'")
    
    area = props['area']
    top_width = props['top_width']
    
    # Handle zero discharge case
    if discharge == 0:
        # For non-rectangular channels, calculate first moment of area about water surface
        # This requires knowledge of the centroid position
        # For rectangular channels, this is simply depth²/2 * width
        # For now, we'll use a simplified approach
        return area * depth / (2 * top_width)
    
    # Calculate momentum function components
    
    # First term: first moment of area about water surface divided by top width
    # For rectangular channels, this is y²/2
    hydrostatic_term = area * depth / (2 * top_width)
    
    # Second term: momentum flux term
    if area <= 0:
        momentum_term = 0
    else:
        momentum_term = discharge**2 / (basic.GRAVITY * area * top_width)
    
    return hydrostatic_term + momentum_term

def sequent_depth(depth1: float, froude1: float) -> float:
    """
    Calculate sequent depth after a hydraulic jump in a rectangular channel.
    
    Parameters:
        depth1 (float): Initial depth before jump (m)
        froude1 (float): Froude number before jump
        
    Returns:
        float: Sequent depth after jump (m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth1 <= 0:
        raise ValueError("Initial depth must be positive")
    if froude1 <= 1:
        raise ValueError("Froude number must be greater than 1 for a hydraulic jump")
    
    # Calculate sequent depth using momentum equation
    # y2/y1 = 0.5 * (sqrt(1 + 8*Fr1²) - 1)
    # This formula is derived from momentum conservation across the jump
    depth_ratio = 0.5 * (math.sqrt(1 + 8 * froude1**2) - 1)
    depth2 = depth1 * depth_ratio
    
    return depth2

def hydraulic_jump_energy_loss(depth1: float, depth2: float) -> float:
    """
    Calculate energy loss across a hydraulic jump.
    
    Parameters:
        depth1 (float): Initial depth before jump (m)
        depth2 (float): Sequent depth after jump (m)
        
    Returns:
        float: Energy loss (m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth1 <= 0 or depth2 <= 0:
        raise ValueError("Depths must be positive")
    if depth2 <= depth1:
        raise ValueError("Sequent depth must be greater than initial depth for a hydraulic jump")
    
    # Calculate energy loss using the formula:
    # ΔE = (y2-y1)³/(4*y1*y2)
    # This is derived from energy equation applied across the jump
    return (depth2 - depth1)**3 / (4 * depth1 * depth2)

def specific_force(depth: float, discharge: float, width: float) -> float:
    """
    Calculate specific force (momentum flux) at a rectangular channel section.
    
    Parameters:
        depth (float): Water depth (m)
        discharge (float): Flow rate (m³/s)
        width (float): Channel width (m)
        
    Returns:
        float: Specific force (N/m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth < 0:  # Allow zero depth for dry channel
        raise ValueError("Depth must be non-negative")
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if width <= 0:
        raise ValueError("Width must be positive")
    
    # For a rectangular channel, the specific force is:
    # F = γy²/2 + ρQ²/(wA)
    # where:
    # - γ is the specific weight of water (ρg)
    # - ρ is water density
    # - A is the flow area
    # - w is the channel width
    
    # Handle zero depth case
    if depth == 0:
        return 0  # No force when depth is zero
    
    # Calculate area
    area = depth * width
    
    # Calculate hydrostatic force component
    # For a rectangular channel, the centroid is at y/2 from the bottom
    gamma = basic.WATER_DENSITY * basic.GRAVITY  # Specific weight of water
    hydrostatic = gamma * depth**2 / 2  # Force per unit width
    
    # Calculate momentum flux component
    if discharge == 0:
        momentum_flux = 0
    else:
        momentum_flux = basic.WATER_DENSITY * discharge**2 / area
    
    # Return specific force (force per unit width)
    return hydrostatic + momentum_flux / width

def specific_force_general(depth: float, discharge: float, 
                          section_props_func: Callable, *args, **kwargs) -> float:
    """
    Calculate specific force (momentum flux) for any channel shape.
    
    Parameters:
        depth (float): Water depth (m)
        discharge (float): Flow rate (m³/s)
        section_props_func: Function that returns section properties given a depth
        *args, **kwargs: Additional arguments to pass to section_props_func
        
    Returns:
        float: Specific force (N/m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth < 0:  # Allow zero depth for dry channel
        raise ValueError("Depth must be non-negative")
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    
    # Handle zero depth case
    if depth == 0:
        return 0  # No force when depth is zero
    
    # Get section properties
    props = section_props_func(depth, *args, **kwargs)
    
    # Check required properties
    required_props = ['area', 'top_width']
    for prop in required_props:
        if prop not in props:
            raise ValueError(f"Section properties function must return '{prop}'")
    
    area = props['area']
    top_width = props['top_width']
    
    # For non-rectangular channels, we need to calculate the first moment of area
    # This is an approximation using a simplified approach
    # For more accuracy, section_props_func should return the first moment
    # directly for the specific channel shape
    
    gamma = basic.WATER_DENSITY * basic.GRAVITY  # Specific weight of water
    
    # Simplified hydrostatic force calculation
    # Assuming a hydrostatic pressure distribution
    hydrostatic = gamma * area * depth / 2  # Approximation for general channels
    
    # Calculate momentum flux component
    if discharge == 0 or area == 0:
        momentum_flux = 0
    else:
        momentum_flux = basic.WATER_DENSITY * discharge**2 / area
    
    # Return specific force (force per unit width of water surface)
    return (hydrostatic + momentum_flux) / top_width