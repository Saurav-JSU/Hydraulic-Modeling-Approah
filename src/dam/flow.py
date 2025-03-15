"""
Dam flow calculations.

This module provides functions for calculating flow over dams,
head-discharge relationships, and energy dissipation.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable

from ..hydraulics import basic
from .geometry import Dam, BroadCrestedWeir, SharpCrestedWeir, OgeeWeir

def calculate_flow_over_dam(dam: Dam, upstream_elevation: float, 
                           downstream_elevation: float,
                           width: float) -> Dict[str, float]:
    """
    Calculate flow characteristics over a dam.
    
    Parameters:
        dam (Dam): Dam object
        upstream_elevation (float): Upstream water surface elevation (m)
        downstream_elevation (float): Downstream water surface elevation (m)
        width (float): Width of dam perpendicular to flow (m)
        
    Returns:
        Dict[str, float]: Dictionary with flow characteristics
    """
    if upstream_elevation <= dam.crest_elevation:
        # No flow over dam
        return {
            'discharge': 0,
            'head': 0,
            'velocity': 0,
            'overflow_area': 0,
            'froude': 0,
            'flow_type': 'No flow'
        }
    
    # Calculate head
    head = upstream_elevation - dam.crest_elevation
    
    # Check for submergence
    submergence = False
    submergence_ratio = 0
    
    if downstream_elevation > dam.crest_elevation:
        submergence = True
        submergence_ratio = (downstream_elevation - dam.crest_elevation) / head
    
    # Calculate discharge
    if isinstance(dam, BroadCrestedWeir):
        discharge = dam.calculate_discharge(head, width)
    elif isinstance(dam, SharpCrestedWeir):
        discharge = dam.calculate_discharge(head, width)
    elif isinstance(dam, OgeeWeir):
        discharge = dam.calculate_discharge(head, width)
    else:
        # Generic calculation for unknown dam types
        # Use a typical discharge coefficient of 0.6
        Cd = 0.6
        g = 9.81
        discharge = Cd * width * head**(3/2) * math.sqrt(2 * g)
    
    # Adjust for submergence if applicable
    if submergence:
        # Apply reduction factor based on submergence ratio
        # Different formulas exist, using a simple one here
        if submergence_ratio < 0.7:
            # Minimal impact on flow
            reduction_factor = 1.0
        elif submergence_ratio < 0.9:
            # Linear reduction
            reduction_factor = 1.0 - 2.0 * (submergence_ratio - 0.7)
        else:
            # Severe reduction
            reduction_factor = 0.6 - (submergence_ratio - 0.9) * 0.6 / 0.1
            reduction_factor = max(0.1, reduction_factor)
        
        discharge *= reduction_factor
    
    # Calculate overflow area
    overflow_area = dam.get_overflow_area(upstream_elevation, width)
    
    # Calculate average velocity
    if overflow_area > 0:
        velocity = discharge / overflow_area
    else:
        velocity = 0
    
    # Calculate Froude number at crest
    if head > 0:
        froude = velocity / math.sqrt(basic.GRAVITY * head)
    else:
        froude = 0
    
    # Determine flow type
    if submergence and submergence_ratio > 0.8:
        flow_type = 'Submerged'
    elif froude < 0.8:
        flow_type = 'Subcritical'
    elif froude > 1.2:
        flow_type = 'Supercritical'
    else:
        flow_type = 'Critical'
    
    return {
        'discharge': discharge,
        'head': head,
        'velocity': velocity,
        'overflow_area': overflow_area,
        'froude': froude,
        'flow_type': flow_type,
        'submergence': submergence,
        'submergence_ratio': submergence_ratio if submergence else 0
    }

def create_rating_curve(dam: Dam, width: float, elevation_range: Tuple[float, float], 
                        num_points: int = 20) -> Dict[str, np.ndarray]:
    """
    Create a rating curve for a dam.
    
    Parameters:
        dam (Dam): Dam object
        width (float): Width of dam perpendicular to flow (m)
        elevation_range (Tuple[float, float]): Min and max water elevations (m)
        num_points (int): Number of points in the curve
        
    Returns:
        Dict[str, np.ndarray]: Rating curve with elevations and discharges
    """
    # Generate range of elevations
    min_elev, max_elev = elevation_range
    
    if min_elev < dam.crest_elevation:
        min_elev = dam.crest_elevation
    
    elevations = np.linspace(min_elev, max_elev, num_points)
    
    # Calculate discharge for each elevation
    discharges = np.zeros_like(elevations)
    heads = np.zeros_like(elevations)
    velocities = np.zeros_like(elevations)
    froude_numbers = np.zeros_like(elevations)
    
    for i, elev in enumerate(elevations):
        # Assume free flow (no submergence)
        flow_result = calculate_flow_over_dam(dam, elev, dam.base_elevation, width)
        
        discharges[i] = flow_result['discharge']
        heads[i] = flow_result['head']
        velocities[i] = flow_result['velocity']
        froude_numbers[i] = flow_result['froude']
    
    return {
        'elevations': elevations,
        'discharges': discharges,
        'heads': heads,
        'velocities': velocities,
        'froude_numbers': froude_numbers
    }

def calculate_energy_dissipation(dam: Dam, upstream_elevation: float, 
                                downstream_elevation: float,
                                width: float) -> Dict[str, float]:
    """
    Calculate energy dissipation for flow over a dam.
    
    Parameters:
        dam (Dam): Dam object
        upstream_elevation (float): Upstream water surface elevation (m)
        downstream_elevation (float): Downstream water surface elevation (m)
        width (float): Width of dam perpendicular to flow (m)
        
    Returns:
        Dict[str, float]: Energy dissipation characteristics
    """
    # Calculate flow characteristics
    flow_result = calculate_flow_over_dam(dam, upstream_elevation, downstream_elevation, width)
    
    # If no flow, return zeros
    if flow_result['discharge'] == 0:
        return {
            'upstream_energy': upstream_elevation,
            'downstream_energy': downstream_elevation,
            'energy_loss': 0,
            'dissipation_ratio': 0,
            'power': 0
        }
    
    # Calculate upstream energy
    discharge = flow_result['discharge']
    upstream_depth = upstream_elevation - dam.base_elevation
    
    # Estimate upstream velocity
    # Assuming an approach channel width equal to dam width
    upstream_area = upstream_depth * width
    upstream_velocity = discharge / upstream_area
    
    # Calculate upstream energy
    upstream_energy = upstream_elevation + (upstream_velocity**2) / (2 * basic.GRAVITY)
    
    # Calculate downstream energy
    downstream_depth = downstream_elevation - dam.base_elevation
    
    # Estimate downstream velocity
    # This is a simplification; in reality, the downstream channel
    # may have different geometry
    downstream_area = downstream_depth * width
    downstream_velocity = discharge / downstream_area
    
    # Calculate downstream energy
    downstream_energy = downstream_elevation + (downstream_velocity**2) / (2 * basic.GRAVITY)
    
    # Calculate energy loss
    energy_loss = upstream_energy - downstream_energy
    
    # Calculate dissipation ratio
    initial_energy = upstream_energy - dam.base_elevation
    if initial_energy > 0:
        dissipation_ratio = energy_loss / initial_energy
    else:
        dissipation_ratio = 0
    
    # Calculate power dissipation
    rho = 1000  # Water density (kg/m³)
    g = 9.81    # Gravitational acceleration (m/s²)
    power = rho * g * discharge * energy_loss
    
    return {
        'upstream_energy': upstream_energy,
        'downstream_energy': downstream_energy,
        'energy_loss': energy_loss,
        'dissipation_ratio': dissipation_ratio,
        'power': power
    }

def estimate_tailwater_profile(dam: Dam, discharge: float, width: float,
                             downstream_slope: float, downstream_roughness: float,
                             distance: float, num_points: int = 50) -> Dict[str, np.ndarray]:
    """
    Estimate the tailwater profile downstream of a dam.
    
    Parameters:
        dam (Dam): Dam object
        discharge (float): Flow rate (m³/s)
        width (float): Width of channel downstream of dam (m)
        downstream_slope (float): Channel bed slope downstream of dam (m/m)
        downstream_roughness (float): Manning's roughness of downstream channel
        distance (float): Distance to calculate profile (m)
        num_points (int): Number of points in the profile
        
    Returns:
        Dict[str, np.ndarray]: Tailwater profile with distances and elevations
    """
    from ..channel.geometry import RectangularChannel
    from ..channel.normal_flow import normal_depth
    from ..channel.gvf import direct_step_method
    
    # Create downstream channel
    downstream_channel = RectangularChannel(bottom_width=width, roughness=downstream_roughness)
    
    # Calculate normal depth in downstream channel
    yn = normal_depth(downstream_channel, discharge, downstream_slope)
    
    # Calculate critical depth
    yc = downstream_channel.critical_depth(discharge)
    
    # Estimate depth immediately downstream of dam
    # This is a simplification and would depend on the actual dam type
    if isinstance(dam, OgeeWeir):
        # Ogee spillways often have dissipation structures
        # that create near-normal depth conditions
        downstream_depth = max(yn, yc)
    else:
        # Other dam types may create supercritical flow
        downstream_depth = 0.9 * yc
    
    # Calculate tailwater profile
    # Using direct step method to solve the GVF equation
    profile = direct_step_method(
        downstream_channel, discharge, downstream_slope,
        downstream_depth, yn, max_distance=distance, distance_step=distance/num_points
    )
    
    # Shift x coordinates to start at dam
    profile['x'] = profile['x']
    
    # Shift z coordinates to match dam base elevation
    z_offset = dam.base_elevation - profile['z'][0]
    profile['z'] += z_offset
    profile['wse'] += z_offset
    
    return profile

def hydraulic_jump_location(dam: Dam, discharge: float, width: float,
                          downstream_slope: float, downstream_roughness: float,
                          tailwater_depth: float) -> Dict[str, float]:
    """
    Estimate the location and characteristics of a hydraulic jump downstream of a dam.
    
    Parameters:
        dam (Dam): Dam object
        discharge (float): Flow rate (m³/s)
        width (float): Width of channel downstream of dam (m)
        downstream_slope (float): Channel bed slope downstream of dam (m/m)
        downstream_roughness (float): Manning's roughness of downstream channel
        tailwater_depth (float): Depth of tailwater at a control point (m)
        
    Returns:
        Dict[str, float]: Hydraulic jump characteristics
    """
    from ..channel.geometry import RectangularChannel
    from ..hydraulics.energy import sequent_depth
    
    # Create downstream channel
    downstream_channel = RectangularChannel(bottom_width=width, roughness=downstream_roughness)
    
    # Calculate critical depth
    yc = downstream_channel.critical_depth(discharge)
    
    # Estimate initial depth at dam toe (supercritical)
    initial_depth = 0.7 * yc
    
    # Calculate initial velocity
    initial_area = initial_depth * width
    initial_velocity = discharge / initial_area
    
    # Calculate initial Froude number
    initial_froude = basic.froude_number(initial_velocity, initial_depth)
    
    # Check if jump is possible
    if initial_froude <= 1:
        # Flow is not supercritical, no jump possible
        return {
            'jump_possible': False,
            'reason': 'Initial flow is not supercritical'
        }
    
    # Calculate sequent depth
    sequent = sequent_depth(initial_depth, initial_froude)
    
    # Check if tailwater depth is sufficient for jump
    if tailwater_depth < sequent * 0.9:
        # Tailwater too low for jump near dam
        # Jump might occur further downstream or be swept away
        return {
            'jump_possible': False,
            'reason': 'Tailwater depth insufficient for jump',
            'sequent_required': sequent,
            'tailwater_actual': tailwater_depth
        }
    
    # Estimate jump location
    # If tailwater matches sequent depth, jump occurs at dam toe
    # If tailwater > sequent, jump is pushed upstream
    # If tailwater < sequent but > 0.9*sequent, jump is pushed downstream
    
    if abs(tailwater_depth - sequent) < 0.05 * sequent:
        # Jump at dam toe
        location = 0
    elif tailwater_depth > sequent:
        # Jump pushed upstream, which means at dam toe for our purposes
        location = 0
    else:
        # Jump pushed downstream
        # This is a simplified estimate; actual location would require
        # solving momentum and energy equations along the channel
        location_factor = (sequent - tailwater_depth) / (0.1 * sequent)
        location = location_factor * 10 * initial_depth  # Rough estimate
    
    # Calculate energy loss in jump
    initial_energy = initial_depth + (initial_velocity**2) / (2 * basic.GRAVITY)
    
    sequent_velocity = discharge / (sequent * width)
    sequent_energy = sequent + (sequent_velocity**2) / (2 * basic.GRAVITY)
    
    energy_loss = initial_energy - sequent_energy
    
    # Classify jump based on Froude number
    if initial_froude < 1.7:
        jump_type = "Undular jump"
    elif initial_froude < 2.5:
        jump_type = "Weak jump"
    elif initial_froude < 4.5:
        jump_type = "Oscillating jump"
    elif initial_froude < 9.0:
        jump_type = "Steady jump"
    else:
        jump_type = "Strong jump"
    
    return {
        'jump_possible': True,
        'location': location,
        'initial_depth': initial_depth,
        'sequent_depth': sequent,
        'initial_velocity': initial_velocity,
        'initial_froude': initial_froude,
        'energy_loss': energy_loss,
        'jump_type': jump_type,
        'depth_ratio': sequent / initial_depth
    }