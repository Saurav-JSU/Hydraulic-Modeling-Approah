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
    # Estimate channel width from cross-section at normal depth
    if channel_slope <= 0:
        raise ValueError("Channel slope must be positive")
    
    yn = normal_depth(channel, discharge, channel_slope)
    width = channel.top_width(yn)
    
    # Calculate upstream water elevation at dam
    upstream_elevation = dam.get_upstream_water_elevation(discharge, width)
    
    # Calculate upstream depth (relative to dam base)
    # This is the hydraulic depth at the dam, which serves as 
    # the boundary condition for the backwater calculation
    upstream_depth = upstream_elevation - dam.base_elevation
    
    # Calculate backwater curve using the GVF (gradually varied flow) module
    # The upstream_depth is the control depth at the downstream end
    backwater = backwater_calculation(
        channel, discharge, channel_slope,
        upstream_depth,  # Use depth relative to channel bed
        control_location='downstream',
        distance=distance,
        num_points=num_points
    )
    
    # Adjust z coordinates to match dam base elevation
    # The backwater calculation typically uses a local coordinate system
    # We need to shift to the global coordinate system based on dam elevation
    # This ensures correct vertical positioning of the water surface profile
    z_offset = dam.base_elevation
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
        # Interpolate using linear interpolation
        area = np.interp(wse, invert_elevations, areas)
        is_extrapolated = False
    else:
        # Extrapolate using last two points
        # This is a linear extrapolation which is a reasonable approximation
        # for small extensions beyond the known data range
        slope = (areas[-1] - areas[-2]) / (invert_elevations[-1] - invert_elevations[-2])
        area = areas[-1] + slope * (wse - invert_elevations[-1])
        is_extrapolated = True
    
    # Calculate reservoir volume using the trapezoidal rule
    # This provides better numerical stability than the previous implementation
    volume = 0
    
    # Find the index where wse would be inserted to maintain sorted order
    # This helps properly segment the calculation for more accurate results
    idx = np.searchsorted(invert_elevations, wse)
    
    # Calculate volume up to the last elevation below wse
    for i in range(1, min(idx, len(invert_elevations))):
        h1 = invert_elevations[i-1]
        h2 = invert_elevations[i]
        a1 = areas[i-1]
        a2 = areas[i]
        
        # Apply trapezoidal rule: V = (h2-h1) * (a1+a2)/2
        segment_volume = (h2 - h1) * (a1 + a2) / 2
        volume += segment_volume
    
    # Calculate final segment from the last elevation below wse to wse itself
    if 0 < idx < len(invert_elevations):
        # wse is between two known elevations
        h1 = invert_elevations[idx-1]
        h2 = wse
        a1 = areas[idx-1]
        # Interpolate area at wse
        a2 = np.interp(wse, [invert_elevations[idx-1], invert_elevations[idx]], 
                       [areas[idx-1], areas[idx]])
        
        # Calculate volume of the final segment
        segment_volume = (h2 - h1) * (a1 + a2) / 2
        volume += segment_volume
    elif idx == len(invert_elevations):
        # wse is above all known elevations, extrapolate the last segment
        h1 = invert_elevations[-1]
        h2 = wse
        a1 = areas[-1]
        # a2 should be the same as the area calculated earlier through extrapolation
        a2 = area
        
        # Calculate volume of the extrapolated segment
        segment_volume = (h2 - h1) * (a1 + a2) / 2
        volume += segment_volume
    
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
    # Calculate reservoir characteristics based on outflow
    # For steady-state conditions, the outflow determines the water surface elevation
    reservoir = calculate_reservoir_characteristics(dam, invert_elevations, areas, outflow, width)
    
    # Extract volume
    volume = reservoir['reservoir_volume']
    
    # Calculate the representative flow rate for residence time calculation
    # In true steady-state conditions, inflow should equal outflow
    # For non-steady conditions:
    #   - If inflow > outflow: reservoir is filling (longer residence time)
    #   - If outflow > inflow: reservoir is draining (shorter residence time)
    # Hydraulically, the appropriate flow to use depends on the application:
    if abs(inflow - outflow) / max(max(inflow, outflow), 1e-6) < 0.05:
        # Flows are within 5% - essentially steady state
        # Use outflow as it controls the water surface elevation
        flow_rate = outflow
    else:
        # Non-steady state - use minimum for conservative residence time
        # This accounts for the fact that some water will stay longer in the reservoir
        flow_rate = min(inflow, outflow)
    
    if flow_rate <= 0:
        # Cannot calculate residence time with zero or negative flow
        return {
            'residence_time_seconds': float('inf'),
            'residence_time_days': float('inf'),
            'reservoir_volume': volume,
            'avg_flow': flow_rate
        }
    
    # Calculate residence time (V/Q) - basic hydraulic principle
    residence_time_seconds = volume / flow_rate
    
    # Convert to days for easier interpretation
    residence_time_days = residence_time_seconds / (24 * 3600)
    
    return {
        'residence_time_seconds': residence_time_seconds,
        'residence_time_days': residence_time_days,
        'reservoir_volume': volume,
        'avg_flow': flow_rate
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
    
    # Adjust x coordinates for spatial continuity
    # For upstream: convert to negative distances with dam at x=0
    # This preserves the channel direction and flow direction
    upstream_x = -np.flip(backwater['x'])  # Negative and flipped for upstream direction
    downstream_x = tailwater['x']  # Positive for downstream direction
    
    # Ensure vertical alignment at the dam
    # The WSE and bed elevation should align properly at the dam location (x=0)
    # No additional shifting needed if calculate_backwater_curve is already aligned
    
    # Combine profiles
    x = np.concatenate([upstream_x, downstream_x])
    z = np.concatenate([backwater['z'], tailwater['z']])
    wse = np.concatenate([backwater['wse'], tailwater['wse']])
    y = np.concatenate([backwater['y'], tailwater['y']])
    v = np.concatenate([backwater['v'], tailwater['v']])
    
    # Add dam profile for visualization
    # Create a dense set of points for dam structure detail
    dam_x = np.linspace(-0.1, 0.1, 10)  # Small x range for dam width
    
    # Get dam profile (structure only)
    dam_profile = dam.get_profile(dam_x)
    
    # Insert dam profile at x=0 (between upstream and downstream sections)
    insert_idx = len(upstream_x)
    
    # Insert dam geometry
    x = np.insert(x, insert_idx, dam_profile['x'])
    z = np.insert(z, insert_idx, dam_profile['z'])
    
    # Add NaN for water surface at dam location
    # This creates a visual break at the dam and prevents connecting
    # the upstream and downstream water surfaces across the dam
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
    # These are fundamental hydraulic parameters that define the flow regime
    yn = normal_depth(channel, discharge, channel_slope)
    yc = channel.critical_depth(discharge)
    
    # Calculate upstream water elevation at dam
    upstream_elevation = dam.get_upstream_water_elevation(discharge, width)
    upstream_depth = upstream_elevation - dam.base_elevation
    
    # Calculate Froude numbers at key sections
    # Froude number < 1: Subcritical flow
    # Froude number > 1: Supercritical flow
    # Froude number = 1: Critical flow
    area_n = channel.area(yn)
    velocity_n = discharge / area_n
    froude_n = basic.froude_number(velocity_n, yn)
    
    area_c = channel.area(yc)
    velocity_c = discharge / area_c
    froude_c = basic.froude_number(velocity_c, yc)  # Should be close to 1
    
    area_u = channel.area(upstream_depth)
    velocity_u = discharge / area_u
    froude_u = basic.froude_number(velocity_u, upstream_depth)
    
    # Determine the controlling section based on hydraulic principles
    # For subcritical flow: control is downstream
    # For supercritical flow: control is upstream
    if froude_n < 1:  # Subcritical flow in channel
        if upstream_depth > yn:
            # Backwater effect: M1 curve (mild slope, backwater)
            # The dam is controlling the upstream water surface
            control_type = "Dam-controlled (M1 profile)"
            control_location = "Downstream (at dam)"
        elif abs(upstream_depth - yn) / yn < 0.01:
            # Within 1% of normal depth - effectively normal flow
            control_type = "Channel-controlled (normal depth)"
            control_location = "Upstream (normal depth)"
        else:
            # Drawdown curve: M2 profile (mild slope, drawdown)
            # Some downstream control is causing drawdown
            control_type = "Dam-controlled (M2 profile)"
            control_location = "Downstream (at dam)"
    else:  # Supercritical flow in channel
        if upstream_depth < yn:
            # S1 curve (steep slope, backwater)
            control_type = "Dam-controlled (S1 profile)"
            control_location = "Downstream (at dam)"
        else:
            # S2 curve or normal depth in supercritical flow
            control_type = "Channel-controlled (normal depth)"
            control_location = "Upstream (normal depth)"
    
    # Determine if critical flow occurs
    # This occurs at hydraulic controls where flow transitions between
    # subcritical and supercritical regimes
    critical_depth_location = "None"
    if froude_n < 1 and froude_u > 1:
        critical_depth_location = "Between normal depth and dam"
    elif froude_n > 1 and froude_u < 1:
        critical_depth_location = "Between normal depth and dam"
    elif abs(froude_u - 1.0) < 0.05:
        # Close to critical flow at dam
        critical_depth_location = "At dam approach"
    
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