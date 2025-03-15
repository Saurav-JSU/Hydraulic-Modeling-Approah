"""
Dam backwater calculations.

This module provides functions for calculating backwater effects
of dams on upstream channels and reservoirs.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable

from ..hydraulics import basic
from ..channel.geometry import Channel
from ..channel.normal_flow import normal_depth
from .geometry import Dam

def calculate_backwater_curve(
    dam: Dam,
    channel: Channel,
    discharge: float,
    channel_slope: float,
    distance: float,
    num_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Calculate the backwater curve upstream of a dam.
    
    Parameters:
        dam (Dam): Dam object
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        channel_slope (float): Channel bed slope (m/m)
        distance (float): Upstream distance to calculate (m)
        num_points (int): Number of points in the curve
        
    Returns:
        Dict[str, np.ndarray]: Backwater curve with distances and elevations
    """
    from ..channel.gvf import backwater_calculation
    
    # Calculate upstream water elevation at dam
    # Assume channel width equals dam width
    # This is a simplification; in reality, the approach channel
    # width might differ from the dam width
    
    # Estimate channel width from cross-section at normal depth
    if channel_slope <= 0:
        raise ValueError("Channel slope must be positive")
    
    yn = normal_depth(channel, discharge, channel_slope)
    width = channel.top_width(yn)
    
    # Calculate upstream water elevation at dam
    upstream_elevation = dam.get_upstream_water_elevation(discharge, width)
    
    # Calculate backwater curve
    # Using backwater calculation from GVF module
    backwater = backwater_calculation(
        channel, discharge, channel_slope,
        upstream_elevation - dam.base_elevation,
        control_location='downstream',
        distance=distance,
        num_points=num_points
    )
    
    # Shift z coordinates to match dam base elevation
    z_offset = dam.base_elevation - backwater['z'][0]
    backwater['z'] += z_offset
    backwater['wse'] += z_offset
    
    return backwater

def calculate_reservoir_characteristics(
    dam: Dam,
    invert_elevations: np.ndarray,
    areas: np.ndarray,
    discharge: float,
    width: float
) -> Dict[str, float]:
    """
    Calculate reservoir characteristics for a given dam and discharge.
    
    Parameters:
        dam (Dam): Dam object
        invert_elevations (np.ndarray): Elevations for stage-storage curve (m)
        areas (np.ndarray): Surface areas at corresponding elevations (m²)
        discharge (float): Flow rate (m³/s)
        width (float): Width of dam perpendicular to flow (m)
        
    Returns:
        Dict[str, float]: Reservoir characteristics
    """
    # Calculate water surface elevation
    wse = dam.get_upstream_water_elevation(discharge, width)
    
    # Ensure reservoir data is sorted by elevation
    sort_idx = np.argsort(invert_elevations)
    invert_elevations = invert_elevations[sort_idx]
    areas = areas[sort_idx]
    
    # Check if water level is within stage-storage curve range
    if wse < invert_elevations[0]:
        # Below lowest elevation in stage-storage curve
        return {
            'water_elevation': wse,
            'reservoir_area': 0,
            'reservoir_volume': 0,
            'is_extrapolated': True
        }
    
    # Find reservoir area at water surface elevation
    if wse <= invert_elevations[-1]:
        # Interpolate
        area = np.interp(wse, invert_elevations, areas)
        is_extrapolated = False
    else:
        # Extrapolate
        # Using last two points for linear extrapolation
        slope = (areas[-1] - areas[-2]) / (invert_elevations[-1] - invert_elevations[-2])
        area = areas[-1] + slope * (wse - invert_elevations[-1])
        is_extrapolated = True
    
    # Calculate reservoir volume
    # Using trapezoidal rule to integrate the stage-storage curve
    
    # First, calculate volume up to the highest point in stage-storage curve
    volume = 0
    for i in range(1, len(invert_elevations)):
        h1 = invert_elevations[i-1]
        h2 = invert_elevations[i]
        a1 = areas[i-1]
        a2 = areas[i]
        
        if h1 >= wse:
            # Above water level, stop calculating
            break
        
        if h2 > wse:
            # Last segment crosses water level, adjust h2 and a2
            ratio = (wse - h1) / (h2 - h1)
            h2 = wse
            a2 = a1 + ratio * (a2 - a1)
        
        # Trapezoidal volume for this segment
        segment_volume = (h2 - h1) * (a1 + a2) / 2
        volume += segment_volume
    
    # If water level is above the highest point in stage-storage curve,
    # add the extrapolated volume
    if wse > invert_elevations[-1]:
        h1 = invert_elevations[-1]
        h2 = wse
        a1 = areas[-1]
        a2 = area
        
        # Trapezoidal volume for extrapolated segment
        extrapolated_volume = (h2 - h1) * (a1 + a2) / 2
        volume += extrapolated_volume
    
    return {
        'water_elevation': wse,
        'reservoir_area': area,
        'reservoir_volume': volume,
        'is_extrapolated': is_extrapolated
    }

def estimate_reservoir_residence_time(
    dam: Dam,
    invert_elevations: np.ndarray,
    areas: np.ndarray,
    inflow: float,
    outflow: float,
    width: float
) -> Dict[str, float]:
    """
    Estimate the hydraulic residence time of water in a reservoir.
    
    Parameters:
        dam (Dam): Dam object
        invert_elevations (np.ndarray): Elevations for stage-storage curve (m)
        areas (np.ndarray): Surface areas at corresponding elevations (m²)
        inflow (float): Inflow rate to reservoir (m³/s)
        outflow (float): Outflow rate from reservoir (m³/s)
        width (float): Width of dam perpendicular to flow (m)
        
    Returns:
        Dict[str, float]: Residence time characteristics
    """
    # Calculate reservoir characteristics
    reservoir = calculate_reservoir_characteristics(dam, invert_elevations, areas, outflow, width)
    
    # Extract volume
    volume = reservoir['reservoir_volume']
    
    # Calculate average flow rate
    avg_flow = (inflow + outflow) / 2
    
    if avg_flow <= 0:
        # Cannot calculate residence time
        return {
            'residence_time_seconds': float('inf'),
            'residence_time_days': float('inf'),
            'reservoir_volume': volume,
            'avg_flow': avg_flow
        }
    
    # Calculate residence time (V/Q)
    residence_time_seconds = volume / avg_flow
    
    # Convert to days
    residence_time_days = residence_time_seconds / (24 * 3600)
    
    return {
        'residence_time_seconds': residence_time_seconds,
        'residence_time_days': residence_time_days,
        'reservoir_volume': volume,
        'avg_flow': avg_flow
    }

def estimate_water_surface_profile(
    dam: Dam,
    channel: Channel,
    discharge: float,
    channel_slope: float,
    upstream_distance: float,
    downstream_distance: float,
    downstream_roughness: float = None,
    num_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Estimate the complete water surface profile including upstream and downstream of dam.
    
    Parameters:
        dam (Dam): Dam object
        channel (Channel): Upstream channel object
        discharge (float): Flow rate (m³/s)
        channel_slope (float): Channel bed slope (m/m)
        upstream_distance (float): Distance to calculate upstream (m)
        downstream_distance (float): Distance to calculate downstream (m)
        downstream_roughness (float, optional): Manning's n for downstream channel
        num_points (int): Total number of points in the profile
        
    Returns:
        Dict[str, np.ndarray]: Complete water surface profile
    """
    from .flow import estimate_tailwater_profile
    
    # Calculate upstream backwater curve
    upstream_points = num_points // 2
    backwater = calculate_backwater_curve(
        dam, channel, discharge, channel_slope, upstream_distance, upstream_points
    )
    
    # Estimate downstream tailwater profile
    downstream_points = num_points - upstream_points
    
    # Use upstream roughness if downstream not specified
    if downstream_roughness is None:
        downstream_roughness = channel.roughness
    
    # Estimate channel width
    yn = normal_depth(channel, discharge, channel_slope)
    width = channel.top_width(yn)
    
    tailwater = estimate_tailwater_profile(
        dam, discharge, width, channel_slope, downstream_roughness,
        downstream_distance, downstream_points
    )
    
    # Adjust x coordinates for tailwater
    # Upstream: negative distances (dam at x=0)
    # Downstream: positive distances
    upstream_x = -np.flip(backwater['x'])
    downstream_x = tailwater['x']
    
    # Adjust z coordinates to ensure continuity at dam
    dam_base = dam.base_elevation
    
    # Combine profiles
    x = np.concatenate([upstream_x, downstream_x])
    z = np.concatenate([backwater['z'], tailwater['z']])
    wse = np.concatenate([backwater['wse'], tailwater['wse']])
    y = np.concatenate([backwater['y'], tailwater['y']])
    v = np.concatenate([backwater['v'], tailwater['v']])
    
    # Add dam profile
    # Create a dense set of points for dam
    dam_x = np.linspace(-0.1, 0.1, 10)
    
    # Get dam profile
    dam_profile = dam.get_profile(dam_x)
    
    # Insert dam profile
    insert_idx = len(upstream_x)
    
    x = np.insert(x, insert_idx, dam_profile['x'])
    z = np.insert(z, insert_idx, dam_profile['z'])
    
    # Add NaN for water surface at dam (will not be plotted)
    wse = np.insert(wse, insert_idx, np.full_like(dam_profile['x'], np.nan))
    y = np.insert(y, insert_idx, np.full_like(dam_profile['x'], np.nan))
    v = np.insert(v, insert_idx, np.full_like(dam_profile['x'], np.nan))
    
    return {
        'x': x,
        'z': z,
        'wse': wse,
        'y': y,
        'v': v
    }

def determine_control_point(
    dam: Dam,
    channel: Channel,
    discharge: float,
    channel_slope: float,
    width: float
) -> Dict[str, float]:
    """
    Determine the control point characteristics for a dam and channel system.
    
    Parameters:
        dam (Dam): Dam object
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        channel_slope (float): Channel bed slope (m/m)
        width (float): Width of dam perpendicular to flow (m)
        
    Returns:
        Dict[str, float]: Control point characteristics
    """
    # Calculate normal and critical depths
    yn = normal_depth(channel, discharge, channel_slope)
    yc = channel.critical_depth(discharge)
    
    # Calculate upstream water elevation at dam
    upstream_elevation = dam.get_upstream_water_elevation(discharge, width)
    upstream_depth = upstream_elevation - dam.base_elevation
    
    # Calculate Froude numbers
    area_n = channel.area(yn)
    velocity_n = discharge / area_n
    froude_n = basic.froude_number(velocity_n, yn)
    
    area_c = channel.area(yc)
    velocity_c = discharge / area_c
    froude_c = basic.froude_number(velocity_c, yc)  # Should be close to 1
    
    area_u = channel.area(upstream_depth)
    velocity_u = discharge / area_u
    froude_u = basic.froude_number(velocity_u, upstream_depth)
    
    # Determine the controlling section
    if froude_n < 1:  # Subcritical flow
        if upstream_depth > yn:
            control_type = "Dam-controlled (M1 profile)"
            control_location = "Downstream (at dam)"
        else:
            control_type = "Channel-controlled (normal depth)"
            control_location = "Upstream (normal depth)"
    else:  # Supercritical flow
        if upstream_depth < yn:
            control_type = "Dam-controlled (S1 profile)"
            control_location = "Downstream (at dam)"
        else:
            control_type = "Channel-controlled (normal depth)"
            control_location = "Upstream (normal depth)"
    
    # Determine if critical flow occurs
    critical_depth_location = "None"
    if froude_n < 1 and froude_u > 1:
        critical_depth_location = "Between normal depth and dam"
    elif froude_n > 1 and froude_u < 1:
        critical_depth_location = "Between normal depth and dam"
    
    return {
        'normal_depth': yn,
        'critical_depth': yc,
        'upstream_depth': upstream_depth,
        'froude_normal': froude_n,
        'froude_critical': froude_c,
        'froude_upstream': froude_u,
        'control_type': control_type,
        'control_location': control_location,
        'critical_depth_location': critical_depth_location
    }