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
        # No flow over dam - water level below crest
        return {
            'discharge': 0,
            'head': 0,
            'velocity': 0,
            'overflow_area': 0,
            'froude': 0,
            'flow_type': 'No flow',
            'submergence': False,
            'submergence_ratio': 0
        }
    
    # Calculate head
    head = upstream_elevation - dam.crest_elevation
    
    # Check for submergence (when downstream water affects flow)
    submergence = False
    submergence_ratio = 0
    
    if downstream_elevation > dam.crest_elevation:
        submergence = True
        submergence_ratio = (downstream_elevation - dam.crest_elevation) / head
    
    # Calculate discharge based on dam type
    if isinstance(dam, BroadCrestedWeir):
        discharge = dam.calculate_discharge(head, width)
    elif isinstance(dam, SharpCrestedWeir):
        discharge = dam.calculate_discharge(head, width)
    elif isinstance(dam, OgeeWeir):
        discharge = dam.calculate_discharge(head, width)
    else:
        # Generic calculation for unknown dam types using standard weir equation
        # Coefficient varies by dam type but 0.6 is a reasonable average value
        # Q = Cd * w * h^(3/2) * sqrt(2g)
        Cd = 0.6
        g = basic.GRAVITY  # Use constant from basic module for consistency
        discharge = Cd * width * head**(3/2) * math.sqrt(2 * g)
    
    # Adjust for submergence if applicable
    # Submergence reduces discharge based on empirical relationships
    if submergence:
        # Using the Villemonte equation for submerged weir flow
        # This is more physically accurate than the previous approach
        # Reduction = [1 - (H2/H1)^n]^0.385
        # Where H2 is downstream head, H1 is upstream head, n varies by weir (1.5 typical)
        
        if isinstance(dam, BroadCrestedWeir):
            n = 1.0  # Empirical exponent for broad-crested weirs
        elif isinstance(dam, SharpCrestedWeir):
            n = 1.5  # Empirical exponent for sharp-crested weirs
        else:
            n = 1.5  # Default for other types including ogee
            
        # Apply Villemonte equation
        if submergence_ratio < 0.95:  # Avoid numerical issues near sr=1
            reduction_factor = (1 - (submergence_ratio)**n)**0.385
            # Add floor to prevent near-zero flow for high submergence
            reduction_factor = max(reduction_factor, 0.05)
            discharge *= reduction_factor
        else:
            # Very high submergence - flow approaches orifice flow
            # Use approximate orifice equation Q = Cd*A*sqrt(2g*Δh)
            delta_h = upstream_elevation - downstream_elevation
            if delta_h > 0:
                orifice_area = width * head  # Approximate opening area
                orifice_coef = 0.6  # Typical for submerged openings
                discharge = orifice_coef * orifice_area * math.sqrt(2 * basic.GRAVITY * delta_h)
            else:
                discharge = 0  # No flow if upstream <= downstream
    
    # Calculate overflow area - cross-sectional area of flow over dam
    overflow_area = dam.get_overflow_area(upstream_elevation, width)
    
    # Calculate average velocity (Q/A) with safeguard for division by zero
    if overflow_area > 0.001:  # Small threshold to avoid numerical issues
        velocity = discharge / overflow_area
    else:
        velocity = 0
    
    # Calculate Froude number at flow control section
    # For a weir, the control section is typically at or near the crest
    # The characteristic length should be the flow depth at this section
    if head > 0:
        # For free-flowing weirs, critical depth occurs near the crest
        # Critical depth is approximately 2/3 of the head for many weirs
        control_depth = 0.67 * head
        # Calculate Froude number using control depth and velocity
        froude = velocity / math.sqrt(basic.GRAVITY * control_depth)
    else:
        froude = 0
    
    # Determine flow type based on hydraulic conditions
    if submergence and submergence_ratio > 0.8:
        # High submergence - flow behaves like an orifice
        flow_type = 'Submerged'
    elif froude < 0.8:
        flow_type = 'Subcritical'
    elif froude > 1.2:
        flow_type = 'Supercritical'
    else:
        # Near critical flow conditions - common at control sections
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
    
    # Ensure min_elev is not less than crest (no flow below crest)
    if min_elev < dam.crest_elevation:
        min_elev = dam.crest_elevation
    
    # Check if max_elev is valid (not below crest)
    if max_elev < dam.crest_elevation:
        # Create arrays with zero discharge values
        # This preserves API compatibility when invalid ranges are provided
        elevations = np.linspace(dam.crest_elevation, dam.crest_elevation + 0.1, num_points)
        discharges = np.zeros_like(elevations)
        heads = np.zeros_like(elevations)
        velocities = np.zeros_like(elevations)
        froude_numbers = np.zeros_like(elevations)
    else:
        # Generate elevation range and allocate arrays
        elevations = np.linspace(min_elev, max_elev, num_points)
        discharges = np.zeros_like(elevations)
        heads = np.zeros_like(elevations)
        velocities = np.zeros_like(elevations)
        froude_numbers = np.zeros_like(elevations)
        
        # Calculate discharge for each elevation
        for i, elev in enumerate(elevations):
            # Assume free flow (no submergence) for rating curve
            # This is standard practice for basic rating curves
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
    
    # Extract flow data
    discharge = flow_result['discharge']
    upstream_depth = upstream_elevation - dam.base_elevation
    
    # Calculate upstream velocity using continuity equation
    # First estimate upstream channel cross-sectional area
    # For reliability, use a minimum depth to avoid division by near-zero
    safe_upstream_depth = max(upstream_depth, 0.01)  # 1cm minimum depth
    upstream_area = safe_upstream_depth * width
    
    # Calculate upstream velocity (Q/A)
    upstream_velocity = discharge / upstream_area
    
    # Calculate upstream specific energy (depth + velocity head)
    # E = y + v²/2g (specific energy equation)
    upstream_energy = upstream_elevation + (upstream_velocity**2) / (2 * basic.GRAVITY)
    
    # Calculate downstream energy
    downstream_depth = downstream_elevation - dam.base_elevation
    
    # Use a minimum downstream depth to avoid division by zero
    # This is physically reasonable as real flows have minimum depths
    safe_downstream_depth = max(downstream_depth, 0.01)  # 1cm minimum
    
    # Calculate downstream area and velocity
    downstream_area = safe_downstream_depth * width
    downstream_velocity = discharge / downstream_area
    
    # Calculate downstream specific energy
    downstream_energy = downstream_elevation + (downstream_velocity**2) / (2 * basic.GRAVITY)
    
    # Calculate energy loss (difference in specific energy)
    energy_loss = upstream_energy - downstream_energy
    
    # Ensure energy loss is non-negative (physical constraint)
    # Small negative values might occur due to numerical issues
    energy_loss = max(energy_loss, 0)
    
    # Calculate dissipation ratio (energy loss / initial energy above datum)
    initial_energy = upstream_energy - dam.base_elevation
    if initial_energy > 0.01:  # Avoid division by very small numbers
        dissipation_ratio = energy_loss / initial_energy
    else:
        dissipation_ratio = 0
    
    # Calculate power dissipation (P = ρgQH)
    rho = 1000  # Water density (kg/m³)
    g = basic.GRAVITY  # Gravitational acceleration (m/s²)
    power = rho * g * discharge * energy_loss  # Power in Watts
    
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
    
    # Create downstream channel (rectangular assumption)
    downstream_channel = RectangularChannel(bottom_width=width, roughness=downstream_roughness)
    
    # Calculate normal depth in downstream channel using Manning's equation
    yn = normal_depth(downstream_channel, discharge, downstream_slope)
    
    # Calculate critical depth using energy principles
    yc = downstream_channel.critical_depth(discharge)
    
    # Estimate initial depth immediately downstream of dam
    # This is based on hydraulic principles for different dam types
    if isinstance(dam, OgeeWeir):
        # Check if the stilling basin attribute exists before using it
        has_stilling_basin = hasattr(dam, 'has_stilling_basin') and dam.has_stilling_basin
        
        if has_stilling_basin:
            # Stilling basins typically create subcritical flow conditions
            downstream_depth = max(yn, 1.1 * yc)
        else:
            # Standard ogee without stilling basin - typically supercritical
            downstream_depth = min(0.8 * yc, 0.9 * yn)
    elif isinstance(dam, BroadCrestedWeir):
        # Broad-crested weirs typically have less jet contraction
        downstream_depth = min(0.85 * yc, 0.95 * yn)
    elif isinstance(dam, SharpCrestedWeir):
        # Sharp-crested weirs have significant jet contraction
        # Initial depth is smaller due to contracted jet
        downstream_depth = min(0.65 * yc, 0.85 * yn)
    else:
        # Generic default - assume moderate contraction
        downstream_depth = min(0.75 * yc, 0.9 * yn)
    
    # Ensure depth is physically reasonable (positive value)
    downstream_depth = max(downstream_depth, 0.05)  # 5cm minimum depth
    
    # Calculate tailwater profile using direct step method
    # This solves gradually varied flow equations numerically
    profile = direct_step_method(
        downstream_channel, discharge, downstream_slope,
        downstream_depth, yn, max_distance=distance, distance_step=distance/num_points
    )
    
    # Ensure x coordinates start at dam (x=0)
    profile['x'] = np.linspace(0, distance, len(profile['x']))
    
    # Adjust elevations to match dam base elevation for proper alignment
    # This ensures vertical continuity in the complete water surface profile
    z_offset = dam.base_elevation
    profile['z'] = dam.base_elevation - downstream_slope * profile['x']  # Linear bed profile
    profile['wse'] = profile['z'] + profile['y']  # WSE = bed + depth
    
    return profile

def hydraulic_jump_location(dam: Dam, discharge: float, width: float,
                          downstream_slope: float, downstream_roughness: float,
                          tailwater_depth: float) -> Dict[str, float]:
    """
    Estimate the location and characteristics of a hydraulic jump downstream of a dam.
    """
    from ..channel.geometry import RectangularChannel
    from ..hydraulics.energy import sequent_depth
    
    # Create downstream channel model
    downstream_channel = RectangularChannel(bottom_width=width, roughness=downstream_roughness)
    
    # Calculate critical depth using energy principles
    yc = downstream_channel.critical_depth(discharge)
    
    # Calculate initial supercritical depth at dam toe based on dam type and physics
    # This uses energy principles and empirical knowledge of different dam designs
    
    # Determine relative height (H/yc) - important parameter for spillway hydraulics
    relative_height = dam.height / yc if yc > 0 else float('inf')
    
    # Determine energy dissipation characteristics based on dam type
    if isinstance(dam, OgeeWeir):
        # Check if the stilling basin attribute exists before using it
        has_stilling_basin = hasattr(dam, 'has_stilling_basin') and dam.has_stilling_basin
        
        if has_stilling_basin:
            # With stilling basin, flow is often forced to jump within basin
            # Return special case for stilling basin
            return {
                'jump_possible': True,
                'location': 0.5,  # Typically in middle of stilling basin
                'initial_depth': 0.6 * yc,  # Approximation for entrance depth
                'sequent_depth': 1.4 * yc,  # Approximation for basin depth
                'initial_velocity': discharge / (0.6 * yc * width),
                'initial_froude': 2.7,  # Typical design value for stilling basins
                'energy_loss': 0.3 * (dam.height + 1.5 * yc),  # Approximation
                'jump_type': "Designed hydraulic jump in stilling basin",
                'depth_ratio': 2.3,  # Typical for designed basins
                'momentum_before': 0,  # Placeholder
                'momentum_after': 0   # Placeholder
            }
        else:
            # Standard ogee without stilling basin
            # Energy loss coefficient based on design quality
            energy_loss_coef = 0.2  # Typical for well-designed ogee
            # Initial depth related to critical depth
            # For high dams, jet effects reduce depth further
            if relative_height > 10:
                # High dam with significant fall
                toe_depth_factor = 0.5  # Higher energy, more contraction
            else:
                # Lower dam
                toe_depth_factor = 0.6

    elif isinstance(dam, BroadCrestedWeir):
        # Broad-crested weirs have less energy dissipation
        energy_loss_coef = 0.3
        toe_depth_factor = 0.7  # Less contraction than ogee
    else:  # Sharp-crested or other
        # Sharp crests create more turbulence and air entrainment
        energy_loss_coef = 0.4
        toe_depth_factor = 0.55  # More contraction than broad-crested
    
    # Calculate initial depth at dam toe based on factors above
    initial_depth = toe_depth_factor * yc
    
    # Ensure minimum physical depth
    initial_depth = max(initial_depth, 0.02)  # 2cm minimum depth
    
    # Calculate initial velocity using continuity equation (Q = VA)
    initial_area = initial_depth * width
    initial_velocity = discharge / initial_area
    
    # Calculate initial Froude number using definition: Fr = v/√(gy)
    initial_froude = basic.froude_number(initial_velocity, initial_depth)
    
    # Check if jump is possible - need supercritical flow (Fr > 1)
    if initial_froude <= 1.05:  # Allow small tolerance
        return {
            'jump_possible': False,
            'reason': 'Initial flow is not supercritical (Fr ≤ 1.05)',
            'initial_froude': initial_froude
        }
    
    # Calculate sequent depth using momentum equation
    # This is the required conjugate depth for a jump to occur
    sequent = sequent_depth(initial_depth, initial_froude)
    
    # Check if tailwater depth is sufficient for jump
    # Jump forms when tailwater approaches sequent depth
    if tailwater_depth < sequent * 0.9:
        # Tailwater too low - jump will be pushed far downstream or won't form
        return {
            'jump_possible': False,
            'reason': 'Tailwater depth insufficient for jump',
            'sequent_required': sequent,
            'tailwater_actual': tailwater_depth,
            'initial_froude': initial_froude
        }
    
    # Calculate friction slope using Manning's equation for initial flow
    n = downstream_roughness
    hydraulic_radius = initial_depth  # For wide rectangular channels
    friction_slope = (n * initial_velocity / hydraulic_radius**(2/3))**2
    
    # Calculate location where jump would form based on tailwater conditions
    # This incorporates momentum balance principles with empirical factors
    
    # Minimum physical length for supercritical flow development
    # Based on Froude number (higher Fr = longer development length)
    if initial_froude > 4.0:
        dev_coefficient = 5.0  # Strong supercritical development
    elif initial_froude > 2.0:
        dev_coefficient = 4.0
    else:
        dev_coefficient = 3.0  # Weaker supercritical development
    
    # Calculate minimum development length
    min_development_length = dev_coefficient * initial_depth * initial_froude
    
    # Calculate jump location based on tailwater vs. sequent depth ratio
    tw_ratio = tailwater_depth / sequent
    
    if abs(tw_ratio - 1.0) < 0.05:
        # Nearly matched tailwater - classic jump location
        # Jump forms at the end of the development region
        location = min_development_length
    elif tw_ratio > 1.0:
        # Higher tailwater pushes jump upstream toward dam
        # But physical limitations prevent it from moving upstream of development length
        push_factor = min(0.7, (tw_ratio - 1.0))  # Limited by physical constraints
        location = max(min_development_length * (1.0 - push_factor), 0.7 * min_development_length)
    else:
        # Lower tailwater - jump pushed downstream by momentum balance
        # Calculate based on gradually varied flow principles
        
        # First estimate energy slope needed to achieve required depth change
        energy_deficit = sequent - tailwater_depth
        
        # Effective slope (difference between bed slope and friction slope)
        # affects how quickly depth increases to meet sequent requirement
        effective_slope = abs(friction_slope - downstream_slope)
        
        if effective_slope > 1e-6:  # Ensure not dividing by near-zero
            # Distance needed to achieve required depth increase
            # Uses the principle that depth changes at rate proportional to (S0-Sf)
            distance_factor = energy_deficit / (effective_slope * initial_depth)
            additional_distance = initial_depth * distance_factor
            location = min_development_length + additional_distance
        else:
            # If slopes nearly balance, depth changes very slowly
            # Use a larger default extension
            location = min_development_length * 3.0
    
    # Calculate energy loss in hydraulic jump using before/after specific energy
    # E = y + v²/2g (specific energy equation)
    initial_energy = initial_depth + (initial_velocity**2) / (2 * basic.GRAVITY)
    
    sequent_velocity = discharge / (sequent * width)
    sequent_energy = sequent + (sequent_velocity**2) / (2 * basic.GRAVITY)
    
    energy_loss = initial_energy - sequent_energy
    # Energy loss should be positive by physics (second law of thermodynamics)
    energy_loss = max(energy_loss, 0)
    
    # Calculate momentum for validation (should be conserved across jump)
    # M = ρQ² + ρgAy̅ (momentum equation for rectangular channel)
    # For unit weight of water:
    momentum_before = (initial_depth**2)/2 + (discharge**2)/(basic.GRAVITY * width * initial_depth)
    momentum_after = (sequent**2)/2 + (discharge**2)/(basic.GRAVITY * width * sequent)
    
    # Classify jump using standard Froude number classifications
    if initial_froude < 1.7:
        jump_type = "Undular jump" # Small waves, minimal energy dissipation
    elif initial_froude < 2.5:
        jump_type = "Weak jump"    # Smooth surface roller, low efficiency
    elif initial_froude < 4.5:
        jump_type = "Oscillating jump" # Oscillating jet, moderate efficiency
    elif initial_froude < 9.0:
        jump_type = "Steady jump"  # Stable jump, good energy dissipation
    else:
        jump_type = "Strong jump"  # Intense turbulence, high energy dissipation
    
    return {
        'jump_possible': True,
        'location': location,
        'initial_depth': initial_depth,
        'sequent_depth': sequent,
        'initial_velocity': initial_velocity,
        'initial_froude': initial_froude,
        'energy_loss': energy_loss,
        'jump_type': jump_type,
        'depth_ratio': sequent / initial_depth,
        'momentum_before': momentum_before,
        'momentum_after': momentum_after
    }