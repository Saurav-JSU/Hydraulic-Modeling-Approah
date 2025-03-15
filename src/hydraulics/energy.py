"""
Energy and momentum principles for open channel flow analysis.

This module contains functions for analyzing energy and momentum in open channels,
including specific energy, momentum functions, and hydraulic jump calculations.
"""

import math
from typing import Dict, Optional, Tuple
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
    Calculate specific energy from discharge.
    
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
    
    # Calculate velocity
    velocity = discharge / (depth * width)
    
    # Calculate specific energy
    return specific_energy(depth, velocity)

def critical_specific_energy(discharge: float, width: float) -> float:
    """
    Calculate minimum specific energy (at critical depth).
    
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
    
    # Calculate critical depth
    yc = basic.critical_depth(discharge, width)
    
    # At critical depth, E = 3/2 * yc
    return 1.5 * yc

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
    Calculate specific momentum function.
    
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
    
    # Calculate momentum function
    # M = y²/2 + q²/(gy)
    # where q = Q/W (discharge per unit width)
    q = discharge / width
    return (depth**2 / 2) + (q**2 / (basic.GRAVITY * depth))

def sequent_depth(depth1: float, froude1: float) -> float:
    """
    Calculate sequent depth after a hydraulic jump.
    
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
    depth_ratio = 0.5 * (math.sqrt(1 + 8 * froude1**2) - 1)
    return depth1 * depth_ratio

def specific_force(depth: float, discharge: float, width: float) -> float:
    """
    Calculate specific force (momentum flux) at a channel section.
    
    Parameters:
        depth (float): Water depth (m)
        discharge (float): Flow rate (m³/s)
        width (float): Channel width (m)
        
    Returns:
        float: Specific force (N/m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if depth <= 0:
        raise ValueError("Depth must be positive")
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if width <= 0:
        raise ValueError("Width must be positive")
    
    # Calculate specific force
    # F = ρgA(ȳ) + ρQ²/A
    # where ȳ is the centroid depth (distance from water surface to centroid)
    
    # For rectangular channel, centroid is at y/2 from surface
    area = depth * width
    centroid_depth = depth / 2
    
    # Calculate hydrostatic force component
    hydrostatic = basic.WATER_DENSITY * basic.GRAVITY * area * centroid_depth
    
    # Calculate momentum flux component
    if depth == 0 or area == 0:
        momentum_flux = 0
    else:
        momentum_flux = basic.WATER_DENSITY * discharge**2 / area
    
    return (hydrostatic + momentum_flux) / width  # Normalize by width for specific force