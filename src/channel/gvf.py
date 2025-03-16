"""
Gradually varied flow calculations for open channels.

This module provides functions for analyzing gradually varied flow (GVF)
and calculating water surface profiles in open channels.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable

from ..hydraulics import basic
from .geometry import Channel
from .normal_flow import normal_depth, critical_depth

def classify_channel_slope(channel: Channel, discharge: float, slope: float) -> str:
    """
    Classify channel slope based on normal and critical depths.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        
    Returns:
        str: Slope classification ('Mild', 'Steep', 'Critical', 'Horizontal', or 'Adverse')
    """
    if discharge <= 0:
        raise ValueError("Discharge must be positive")
    
    if slope < 0:
        return "Adverse"
    
    if slope == 0:
        return "Horizontal"
    
    # Calculate normal and critical depths
    yn = normal_depth(channel, discharge, slope)
    yc = critical_depth(channel, discharge)
    
    # Classification based on comparison
    if abs(yn - yc) < 0.001 * yc:  # Within 0.1% tolerance
        return "Critical"
    elif yn > yc:
        return "Mild"
    else:
        return "Steep"

def classify_flow_profile(channel: Channel, discharge: float, slope: float, 
                         depth: float) -> str:
    """
    Classify water surface profile type.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        depth (float): Water depth (m)
        
    Returns:
        str: Profile classification (M1, M2, M3, S1, S2, S3, C1, C3, H2, H3, A2, A3)
    """
    if discharge <= 0:
        raise ValueError("Discharge must be positive")
    if depth <= 0:
        raise ValueError("Depth must be positive")
    
    # Get slope type
    slope_type = classify_channel_slope(channel, discharge, slope)
    
    # Calculate normal and critical depths
    if slope > 0:
        yn = normal_depth(channel, discharge, slope)
    else:
        yn = float('inf')  # Normal depth doesn't exist for adverse slope
    
    yc = critical_depth(channel, discharge)
    
    # Classification based on slope type and depth
    if slope_type == "Mild":
        if depth > yn:
            return "M1"
        elif yn > depth > yc:
            return "M2"
        else:  # depth < yc
            return "M3"
    
    elif slope_type == "Steep":
        if depth > yc:
            return "S1"
        elif yc > depth > yn:
            return "S2"
        else:  # depth < yn
            return "S3"
    
    elif slope_type == "Critical":
        if depth > yc:  # Note: yc = yn for critical slope
            return "C1"
        else:  # depth < yc
            return "C3"
    
    elif slope_type == "Horizontal":
        if depth > yc:
            return "H2"
        else:  # depth < yc
            return "H3"
    
    else:  # Adverse
        if depth > yc:
            return "A2"
        else:  # depth < yc
            return "A3"

def compute_froude_number(channel: Channel, discharge: float, depth: float) -> float:
    """
    Compute Froude number for a given channel, discharge, and depth.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        depth (float): Water depth (m)
        
    Returns:
        float: Froude number
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if depth <= 0:
        raise ValueError("Depth must be positive")
    
    area = channel.area(depth)
    velocity = discharge / area
    hydraulic_depth = channel.hydraulic_depth(depth)
    
    return velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)

def gvf_equation(channel: Channel, discharge: float, slope: float, depth: float) -> float:
    """
    Compute the right-hand side of the GVF equation dy/dx = (S0 - Sf) / (1 - Fr²).
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        depth (float): Water depth (m)
        
    Returns:
        float: Rate of change of depth with distance (dy/dx)
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if depth <= 0:
        raise ValueError("Depth must be positive")
    
    # Calculate Froude number
    froude = compute_froude_number(channel, discharge, depth)
    
    # Calculate friction slope using Manning's equation
    area = channel.area(depth)
    hydraulic_radius = channel.hydraulic_radius(depth)
    velocity = discharge / area
    
    # Manning's equation for friction slope: Sf = (nV)²/R^(4/3)
    n = channel.roughness
    friction_slope = (n * velocity)**2 / hydraulic_radius**(4/3)
    
    # GVF equation
    denominator = 1 - froude**2
    
    # Improved handling for near-critical flow - smoother transition
    # Physical reasoning: near critical flow, depth changes more rapidly but not infinitely
    if abs(denominator) < 1e-6:
        # Calculate sign based on if flow is slightly subcritical or supercritical
        sign = -1 if denominator < 0 else 1
        # Scale factor based on proximity to critical - smoother transition
        scale_factor = 1e4 * (1e-6 / max(abs(denominator), 1e-10))
        # Direction based on energy slope (S0 - Sf)
        direction = 1 if slope > friction_slope else -1
        return direction * sign * min(scale_factor, 1e6)
    
    return (slope - friction_slope) / denominator

def direct_step_method(
    channel: Channel, 
    discharge: float, 
    slope: float, 
    initial_depth: float, 
    target_depth: float, 
    max_distance: float = 10000, 
    distance_step: float = 10,
    max_depth_step: float = 0.1,
    min_depth_step: float = 0.001,
    convergence_tolerance: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Calculate water surface profile using direct step method.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        initial_depth (float): Starting water depth (m)
        target_depth (float): Target water depth (m) or None
        max_distance (float): Maximum distance to compute (m)
        distance_step (float): Initial step size for distance (m)
        max_depth_step (float): Maximum step size for depth (m)
        min_depth_step (float): Minimum step size for depth (m)
        convergence_tolerance (float): Tolerance for convergence checks
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with distance, depth, and other arrays
    """
    if discharge <= 0:
        raise ValueError("Discharge must be positive")
    if initial_depth <= 0 or target_depth <= 0:
        raise ValueError("Depths must be positive")
    
    # Determine critical conditions
    yc = critical_depth(channel, discharge)
    
    if slope > 0:
        yn = normal_depth(channel, discharge, slope)
    else:
        yn = float('inf')  # No normal depth for adverse or horizontal slopes
    
    # Get profile type for more physically-based direction determination
    profile_type = classify_flow_profile(channel, discharge, slope, initial_depth)
    
    # Determine flow direction based on profile classification
    # Physical reasoning: M1, S1, H2, A2 are backwater curves, M3, S3, H3, A3 are drawdown curves
    # These should be computed in downstream direction
    # M2, S2 curves are typically computed upstream
    if profile_type in ['M1', 'S1', 'C1', 'H2', 'A2', 'M3', 'S3', 'C3', 'H3', 'A3']:
        direction = 1  # Downstream direction
    else:
        # M2, S2 profiles typically computed upstream
        direction = -1  # Upstream direction
    
    # Initialize arrays
    distances = [0]
    depths = [initial_depth]
    velocities = []
    froude_numbers = []
    energy_heads = []
    
    # Calculate initial state
    area = channel.area(initial_depth)
    velocity = discharge / area
    froude = compute_froude_number(channel, discharge, initial_depth)
    
    velocities.append(velocity)
    froude_numbers.append(froude)
    
    # Velocity head
    velocity_head = velocity**2 / (2 * basic.GRAVITY)
    energy_head = initial_depth + velocity_head
    energy_heads.append(energy_head)
    
    # Set up for iteration
    current_depth = initial_depth
    current_distance = 0
    
    # Main iteration loop
    while (current_distance < max_distance and 
          abs(current_depth - target_depth) > convergence_tolerance):
        
        # Compute rate of change of depth with distance
        dy_dx = gvf_equation(channel, discharge, slope, current_depth)
        
        # More sophisticated handling near critical and normal depths
        # Physical reasoning: depth changes more gradually near control depths
        approaching_normal = False
        if slope > 0 and abs(current_depth - yn) < 0.05 * yn:
            approaching_normal = True
            # Exponentially reduce step size as we approach normal depth
            normal_factor = math.exp(-10 * abs(current_depth - yn) / yn)
            max_depth_step *= (1 - 0.9 * normal_factor)
        
        approaching_critical = False
        if abs(current_depth - yc) < 0.1 * yc:
            approaching_critical = True
            # Exponentially reduce step size as we approach critical depth
            critical_factor = math.exp(-10 * abs(current_depth - yc) / yc)
            max_depth_step *= (1 - 0.9 * critical_factor)
        
        # Dynamic step size adjustment based on rate of change
        if approaching_normal or approaching_critical:
            depth_step = min_depth_step
        else:
            # Adaptive step size - smaller when dy/dx changes rapidly
            depth_step = min(max_depth_step, 
                           max(min_depth_step, 
                               0.01 / max(abs(dy_dx), 0.0001)))
        
        # Ensure we don't step past the target depth
        if (current_depth < target_depth and current_depth + depth_step > target_depth) or \
           (current_depth > target_depth and current_depth - depth_step < target_depth):
            depth_step = abs(target_depth - current_depth)
        
        # Set the step direction
        if current_depth < target_depth:
            depth_step = abs(depth_step)  # Increasing depth
        else:
            depth_step = -abs(depth_step)  # Decreasing depth
        
        # Calculate next depth
        next_depth = current_depth + depth_step
        
        # Ensure depth is positive and not too small
        if next_depth <= 0:
            next_depth = min_depth_step
        
        # Calculate the distance step
        if abs(dy_dx) < 1e-10:
            # Almost horizontal water surface
            distance_step = direction * max_distance / 100
        else:
            distance_step = direction * depth_step / dy_dx
        
        # Limit extreme distance steps
        if abs(distance_step) > max_distance / 10:
            distance_step = direction * max_distance / 10
        
        # Update distance and depth
        current_distance += distance_step
        current_depth = next_depth
        
        # Calculate hydraulic parameters at new depth
        area = channel.area(current_depth)
        velocity = discharge / area
        froude = compute_froude_number(channel, discharge, current_depth)
        
        # Velocity head
        velocity_head = velocity**2 / (2 * basic.GRAVITY)
        energy_head = current_depth + velocity_head
        
        # Store results
        distances.append(current_distance)
        depths.append(current_depth)
        velocities.append(velocity)
        froude_numbers.append(froude)
        energy_heads.append(energy_head)
        
        # Improved convergence check to detect oscillations
        if len(depths) > 3:
            if ((depths[-1] - target_depth) * (depths[-3] - target_depth) < 0 and
                abs(depths[-1] - depths[-3]) < convergence_tolerance):
                # We're oscillating and close enough - physically reasonable to stop
                break
        
        # Safety check for maximum iterations
        if len(distances) > 1000:  # Prevent infinite loops
            break
    
    # Convert lists to numpy arrays
    distances = np.array(distances)
    depths = np.array(depths)
    velocities = np.array(velocities)
    froude_numbers = np.array(froude_numbers)
    energy_heads = np.array(energy_heads)
    
    # Calculate bed elevation and water surface elevation
    bed_elevations = -slope * distances
    water_surface_elevations = bed_elevations + depths
    
    return {
        'x': distances,
        'y': depths,
        'z': bed_elevations,
        'wse': water_surface_elevations,
        'v': velocities,
        'fr': froude_numbers,
        'energy': energy_heads
    }

def standard_step_method(
    channel: Channel, 
    discharge: float, 
    slope: float, 
    x_stations: np.ndarray,
    downstream_depth: float,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Calculate water surface profile using standard step method.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        x_stations (np.ndarray): Array of x-coordinates (distances) for computations
        downstream_depth (float): Known water depth at downstream end
        max_iterations (int): Maximum number of iterations for each step
        tolerance (float): Convergence tolerance
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with distance, depth, and other arrays
    """
    if discharge <= 0:
        raise ValueError("Discharge must be positive")
    if downstream_depth <= 0:
        raise ValueError("Downstream depth must be positive")
    
    # Sort stations to ensure they are in ascending order
    x_stations = np.sort(x_stations)
    
    # Initialize arrays
    n_stations = len(x_stations)
    depths = np.zeros(n_stations)
    velocities = np.zeros(n_stations)
    froude_numbers = np.zeros(n_stations)
    energy_heads = np.zeros(n_stations)
    
    # Calculate bed elevations (assuming straight line)
    bed_elevations = -slope * x_stations
    
    # Set downstream conditions (station 0)
    depths[0] = downstream_depth
    area = channel.area(downstream_depth)
    velocities[0] = discharge / area
    froude_numbers[0] = compute_froude_number(channel, discharge, downstream_depth)
    
    # Calculate energy head at downstream section
    velocity_head = velocities[0]**2 / (2 * basic.GRAVITY)
    energy_heads[0] = depths[0] + velocity_head + bed_elevations[0]
    
    # Determine flow regime at control section for better initial guesses
    subcritical = froude_numbers[0] < 1
    
    # Iterate from downstream to upstream
    for i in range(1, n_stations):
        # Distance between current and previous section
        delta_x = x_stations[i] - x_stations[i-1]
        
        # Better initial guess based on flow regime and physics
        if subcritical:
            # For subcritical flow, energy tends to follow bed slope
            energy_diff = bed_elevations[i] - bed_elevations[i-1]
            y_guess = depths[i-1] - energy_diff
        else:
            # For supercritical flow, depth changes more rapidly
            # Use a depth that maintains similar Froude number
            y_guess = depths[i-1] * 0.95  # Slight reduction as we move upstream
        
        # Ensure guess is positive
        y_guess = max(y_guess, 0.01)
        
        # Previous iteration value for convergence check
        y_prev = y_guess
        
        # Iterative solution for current section
        for iter_count in range(max_iterations):
            # Calculate hydraulic parameters at guessed depth
            area = channel.area(y_guess)
            velocity = discharge / area
            
            # Calculate energy head at current section
            velocity_head = velocity**2 / (2 * basic.GRAVITY)
            energy_head = y_guess + velocity_head + bed_elevations[i]
            
            # Calculate average friction slope using a more physically accurate method
            # Use average conveyance method for more accurate head loss calculation
            hyd_radius = channel.hydraulic_radius(y_guess)
            conveyance = (area * hyd_radius**(2/3)) / channel.roughness
            
            hyd_radius_prev = channel.hydraulic_radius(depths[i-1])
            area_prev = channel.area(depths[i-1])
            conveyance_prev = (area_prev * hyd_radius_prev**(2/3)) / channel.roughness
            
            # Harmonic mean of conveyance gives better physical representation
            # of average energy dissipation between sections
            if conveyance > 0 and conveyance_prev > 0:
                avg_conveyance = 2 * conveyance * conveyance_prev / (conveyance + conveyance_prev)
            else:
                avg_conveyance = max(conveyance, conveyance_prev) / 2
            
            # Head loss calculation using conveyance method
            head_loss = (discharge**2 / avg_conveyance**2) * delta_x
            
            # Calculated energy head using energy balance equation
            calculated_energy = energy_heads[i-1] - head_loss
            
            # Error in energy balance
            error = energy_head - calculated_energy
            
            # Check convergence
            if abs(error) < tolerance:
                break
            
            # Improved depth adjustment using Newton-Raphson method
            # dError/dy = 1 - Q²/(g*A³)*dA/dy
            dArea_dy = channel.top_width(y_guess)  # dA/dy = top width
            dError_dy = 1 - (discharge**2) / (basic.GRAVITY * area**3) * dArea_dy
            
            # Apply Newton-Raphson with safeguards
            if abs(dError_dy) > 1e-10:
                correction = error / dError_dy
            else:
                # Fallback to simpler method if derivative is near zero
                correction = error * 0.1
            
            # Damping to improve stability
            damping = 0.7
            correction *= damping
            
            # Limit correction size for stability
            max_correction = 0.2 * y_guess
            if abs(correction) > max_correction:
                correction = max_correction * np.sign(correction)
            
            # Update depth guess
            y_guess -= correction
            
            # Ensure depth is positive
            if y_guess <= 0:
                y_guess = 0.01
                
            # Additional convergence check based on change in depth
            if abs(y_guess - y_prev) < tolerance * y_guess:
                break
                
            y_prev = y_guess
        
        # Store results for current station
        depths[i] = y_guess
        area = channel.area(y_guess)
        velocities[i] = discharge / area
        froude_numbers[i] = compute_froude_number(channel, discharge, y_guess)
        energy_heads[i] = energy_head
    
    # Calculate water surface elevations
    water_surface_elevations = bed_elevations + depths
    
    return {
        'x': x_stations,
        'y': depths,
        'z': bed_elevations,
        'wse': water_surface_elevations,
        'v': velocities,
        'fr': froude_numbers,
        'energy': energy_heads
    }

def backwater_calculation(
    channel: Channel, 
    discharge: float, 
    slope: float, 
    control_depth: float,
    control_location: str = 'downstream',
    distance: float = 1000,
    num_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Calculate backwater curve from a specified control point.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        control_depth (float): Known water depth at control point
        control_location (str): Location of control point ('upstream' or 'downstream')
        distance (float): Distance to calculate profile (m)
        num_points (int): Number of points in the profile
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with distance, depth, and other arrays
    """
    if discharge <= 0:
        raise ValueError("Discharge must be positive")
    if control_depth <= 0:
        raise ValueError("Control depth must be positive")
    
    # Calculate normal depth
    if slope > 0:
        yn = normal_depth(channel, discharge, slope)
    else:
        yn = float('inf')
    
    # Calculate critical depth
    yc = critical_depth(channel, discharge)
    
    # Create x stations
    if control_location.lower() == 'downstream':
        x_stations = np.linspace(0, distance, num_points)
        # Use standard step method for computations
        return standard_step_method(channel, discharge, slope, x_stations, control_depth)
    
    elif control_location.lower() == 'upstream':
        x_stations = np.linspace(distance, 0, num_points)
        # For upstream control, reverse the problem
        result = standard_step_method(channel, discharge, slope, np.flip(x_stations), control_depth)
        # Reverse the results back to original orientation
        for key in result:
            result[key] = np.flip(result[key])
        return result
    
    else:
        raise ValueError("Control location must be 'upstream' or 'downstream'")