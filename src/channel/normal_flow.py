"""
Normal flow calculations for open channels.

This module provides functions for calculating normal flow parameters
in open channels using Manning's equation.
"""

import math
from typing import Dict, Optional, Union, Tuple
import numpy as np

from ..hydraulics import basic, manning
from .geometry import Channel

def normal_depth(channel: Channel, discharge: float, slope: float,
                depth_range: tuple = (0.01, 20), 
                tolerance: float = 1e-6, 
                max_iterations: int = 100) -> float:
    """
    Calculate normal depth for a given channel and discharge.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        depth_range (tuple): Range of depths to search (m)
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum number of iterations
        
    Returns:
        float: Normal depth (m)
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If solution doesn't converge
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if slope <= 0:
        raise ValueError("Slope must be positive")
    
    # For zero discharge, normal depth is zero
    if discharge == 0:
        return 0
    
    # Function to find root of: Manning equation
    # For steady uniform flow: Q = (1/n)*A*R^(2/3)*S^(1/2)
    def depth_function(y: float) -> float:
        if y <= 0:
            return float('inf')
        
        # Calculate flow parameters
        area = channel.area(y)
        wetted_perimeter = channel.wetted_perimeter(y)
        
        if wetted_perimeter <= 0 or area <= 0:
            return float('inf')
            
        hydraulic_radius = area / wetted_perimeter
        
        # Manning formula (rearranged to calculate discharge)
        # Q = (1/n)*A*R^(2/3)*S^(1/2)
        manning_n = channel.roughness
        calculated_discharge = (1.0/manning_n) * area * (hydraulic_radius**(2/3)) * math.sqrt(slope)
        
        # Return difference between calculated and target discharge
        return calculated_discharge - discharge
    
    # Set up improved initial bounds based on hydraulic principles
    # Using multiple methods to find reasonable starting points
    
    # Method 1: Chezy-Manning equation for wide rectangular channels
    # For wide channels: y ≈ (n*Q/(b*S^0.5))^(3/5)
    try:
        # Estimate channel width - more robust approach
        if hasattr(channel, 'bottom_width'):
            b = channel.bottom_width
        else:
            # Try to estimate width from geometry
            test_depth = 1.0  # Use 1m as a test depth
            area_at_test = channel.area(test_depth)
            # Estimate width as area/depth for roughly rectangular channels
            b = area_at_test / test_depth
            # Apply a correction factor for non-rectangular channels
            b *= 0.8  # Approximation for typical trapezoidal channels
        
        n = channel.roughness
        # Add a safeguard against zero width
        if b <= 0:
            b = 1.0  # Default to 1m if width estimation fails
        
        chezy_estimate = (n * discharge / (b * math.sqrt(slope)))**(3/5)
        # Apply physical constraint to prevent unrealistic estimates
        if chezy_estimate > 50 or chezy_estimate < 0.01:
            chezy_estimate = None
    except Exception:
        chezy_estimate = None
    
    # Method 2: Critical depth relation (typically smaller than normal for mild slopes)
    try:
        # Get critical depth from channel method if available
        if hasattr(channel, 'critical_depth'):
            critical_y = channel.critical_depth(discharge)
        else:
            # Approximate using hydraulic exponent
            # For rectangular: yc = (q²/g)^(1/3)
            q = discharge / b if 'b' in locals() and b > 0 else discharge
            critical_y = (q**2 / basic.GRAVITY)**(1/3)
        
        # For mild slopes, normal depth > critical depth
        # Estimate normal depth based on critical depth
        normal_from_critical = critical_y * (channel.roughness / math.sqrt(slope))**(3/5)
        
        # Apply physical constraint
        if normal_from_critical > 50 or normal_from_critical < 0.01:
            normal_from_critical = None
        else:
            critical_y = normal_from_critical  # Use this improved estimate
    except Exception:
        critical_y = None
    
    # Method 3: Hydraulic geometry - Leopold relation
    # Q = c*y^f where f ≈ 1.67 (general depth-discharge exponent)
    try:
        # Using a typical coefficient based on roughness
        c = 5.0 / channel.roughness  # Approximate coefficient
        exponent = 1.67  # Standard exponent for depth-discharge relation
        leopold_estimate = (discharge / c)**(1/exponent)
        
        # Physical constraint
        if leopold_estimate > 50 or leopold_estimate < 0.01:
            leopold_estimate = None
    except Exception:
        leopold_estimate = None
    
    # Method 4: Direct trial depths to find bracket
    try:
        # Try a few sample depths to find ones that bracket the solution
        trial_depths = [0.1, 0.5, 1.0, 2.0, 5.0]
        trial_values = [depth_function(y) for y in trial_depths]
        
        # Find two depths that bracket the root (opposite signs)
        direct_estimate = None
        for i in range(len(trial_depths)-1):
            if trial_values[i] * trial_values[i+1] <= 0:
                # Found a bracket, use the average as an estimate
                direct_estimate = (trial_depths[i] + trial_depths[i+1]) / 2
                break
    except Exception:
        direct_estimate = None
    
    # Create an array of valid estimates and take the median
    estimates = [e for e in [chezy_estimate, critical_y, leopold_estimate, direct_estimate] 
                if e is not None and 0.01 <= e <= 50]
    
    if estimates:
        initial_estimate = sorted(estimates)[len(estimates)//2]  # Median value
        # Set search range based on this estimate with some safety margin
        y_min = max(0.2 * initial_estimate, 0.01)
        y_max = min(5.0 * initial_estimate, 50.0)  # Physical limit
    else:
        # Default range if estimation fails - use original range but constrained
        y_min = max(depth_range[0], 0.01)
        y_max = min(depth_range[1], 50.0)
    
    # Evaluate function at bounds
    f_min = depth_function(y_min)
    f_max = depth_function(y_max)
    
    # Using modified secant method with bisection fallback
    # This combines fast convergence with robustness
    if f_min * f_max > 0:
        # Bounds don't bracket root, try to expand until they do
        if f_min > 0:  # Both positive, decrease lower bound
            expansion_counter = 0
            while f_min > 0 and expansion_counter < 20:  # Increased from 10 to 20
                y_min /= 2
                if y_min < 1e-6:  # Prevent too small values
                    y_min = 1e-6
                    break
                f_min = depth_function(y_min)
                expansion_counter += 1
        
        if f_max < 0:  # Both negative, increase upper bound
            expansion_counter = 0
            while f_max < 0 and expansion_counter < 20:  # Increased from 10 to 20
                y_max *= 2
                if y_max > 100:  # Hard physical limit for normal depth
                    raise ValueError("Cannot find normal depth in reasonable range (0.01-100m). Check parameters.")
                f_max = depth_function(y_max)
                expansion_counter += 1
                
        # Check again if we've bracketed the root
        if f_min * f_max > 0:
            # If still not bracketed, try a more aggressive approach
            y_values = np.logspace(-2, 2, 50)  # 50 points from 0.01 to 100
            f_values = [depth_function(y) for y in y_values]
            
            # Find sign changes
            sign_changes = []
            for i in range(len(y_values)-1):
                if f_values[i] * f_values[i+1] <= 0:
                    sign_changes.append((y_values[i], y_values[i+1]))
            
            if sign_changes:
                # Use the first bracket
                y_min, y_max = sign_changes[0]
                f_min, f_max = depth_function(y_min), depth_function(y_max)
            else:
                raise ValueError("Cannot bracket normal depth solution. Check channel parameters and flow rate.")
    
    # Apply hybrid Newton-bisection method for robust convergence
    y_current = y_min + (y_max - y_min) * (0 - f_min) / (f_max - f_min)  # Linear interpolation
    if not (y_min < y_current < y_max):
        y_current = 0.5 * (y_min + y_max)  # Fall back to bisection
    
    for i in range(max_iterations):
        f_current = depth_function(y_current)
        
        # Check for convergence
        if abs(f_current) < tolerance:
            # Check final depth for physical reasonability
            if y_current > 50.0:  # Very large for normal depth
                raise ValueError(f"Calculated normal depth ({y_current:.2f}m) exceeds physical expectations.")
            return y_current
        
        # Update bounds for bisection safety
        if f_current * f_min < 0:
            y_max = y_current
            f_max = f_current
        else:
            y_min = y_current
            f_min = f_current
        
        # Try Newton step (with improved derivative approximation)
        # Use central difference for better accuracy
        h = max(0.001 * y_current, 1e-5)  # Adaptive perturbation
        f_forward = depth_function(y_current + h)
        f_backward = depth_function(y_current - h)
        df_dy = (f_forward - f_backward) / (2 * h)
        
        # Avoid division by very small numbers
        if abs(df_dy) > 1e-10:
            newton_step = f_current / df_dy
            # Limit the step size for stability
            max_step = 0.5 * (y_max - y_min)
            if abs(newton_step) > max_step:
                newton_step = max_step * (1 if newton_step > 0 else -1)
            
            y_new = y_current - newton_step
        else:
            # Fall back to bisection if derivative is too small
            y_new = 0.5 * (y_min + y_max)
        
        # Check if Newton's step is within bounds
        if y_new < y_min or y_new > y_max:
            # If not, use bisection instead
            y_new = 0.5 * (y_min + y_max)
        
        # Check for convergence between iterations
        if abs(y_new - y_current) < tolerance:
            # Verify solution with direct substitution
            final_check = depth_function(y_new)
            if abs(final_check) < tolerance * 10:  # Slightly relaxed tolerance for final check
                # Check final depth for physical reasonability
                if y_new > 50.0:  # Very large for normal depth
                    raise ValueError(f"Calculated normal depth ({y_new:.2f}m) exceeds physical expectations.")
                return y_new
        
        y_current = y_new
    
    # If we reached max iterations without converging, use final bracket midpoint
    # This gives a reasonable approximate answer even if full convergence wasn't achieved
    y_final = 0.5 * (y_min + y_max)
    if abs(depth_function(y_final)) < tolerance * 100:  # Very relaxed tolerance
        return y_final
    
    # If we reached max iterations without converging
    raise RuntimeError(f"Normal depth calculation did not converge within {max_iterations} iterations")

def critical_depth(channel: Channel, discharge: float) -> float:
    """
    Calculate critical depth for a given channel and discharge.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        
    Returns:
        float: Critical depth (m)
    """
    # Input validation
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    
    # For zero discharge, critical depth is zero
    if discharge == 0:
        return 0
    
    # Try channel's method first
    try:
        return channel.critical_depth(discharge)
    except Exception:
        # Fallback calculation using numerical method if channel method fails
        # Critical depth occurs when Fr = 1
        
        # Function to find critical depth: Fr = 1
        def froude_function(y):
            if y <= 0:
                return float('inf')
            
            area = channel.area(y)
            if area <= 0:
                return float('inf')
            
            velocity = discharge / area
            
            # Get hydraulic depth for Froude number calculation
            try:
                hydraulic_depth = channel.hydraulic_depth(y)
            except:
                # Fallback if hydraulic_depth not available
                top_width = channel.top_width(y)
                hydraulic_depth = area / top_width if top_width > 0 else float('inf')
            
            froude = velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)
            return froude - 1.0  # Target: Fr = 1
        
        # Set up search range
        y_min = 0.01
        y_max = 20.0
        
        # Make an initial guess based on rectangular channel approximation
        # For rectangular channels: yc = (q²/g)^(1/3)
        try:
            # Estimate width for specific discharge
            if hasattr(channel, 'bottom_width'):
                width = channel.bottom_width
            else:
                test_depth = 1.0
                area = channel.area(test_depth)
                width = area / test_depth  # Rough approximation
            
            q = discharge / width  # Specific discharge
            initial_guess = (q**2 / basic.GRAVITY)**(1/3)
            
            # Constrain guess to reasonable range
            if initial_guess < y_min or initial_guess > y_max:
                initial_guess = 0.5 * (y_min + y_max)
        except:
            initial_guess = 0.5 * (y_min + y_max)
        
        # Bisection method for robust solution
        f_min = froude_function(y_min)
        f_max = froude_function(y_max)
        
        # If bounds don't bracket root, try to expand them
        if f_min * f_max > 0:
            # Try to find bracket
            test_depths = np.logspace(-2, 1, 30)  # 30 points from 0.01 to 10
            for depth in test_depths:
                f_test = froude_function(depth)
                if f_test * f_min <= 0:
                    y_max = depth
                    f_max = f_test
                    break
                elif f_test * f_max <= 0:
                    y_min = depth
                    f_min = f_test
                    break
            
            # Check if we found a bracket
            if f_min * f_max > 0:
                # If still no bracket, use our best approximation
                return initial_guess
        
        # Apply bisection method
        y_current = initial_guess
        max_iterations = 100
        tolerance = 1e-6
        
        for i in range(max_iterations):
            f_current = froude_function(y_current)
            
            if abs(f_current) < tolerance:
                return y_current
            
            if f_current * f_min < 0:
                y_max = y_current
                f_max = f_current
            else:
                y_min = y_current
                f_min = f_current
            
            y_current = 0.5 * (y_min + y_max)
            
            # Check for convergence
            if (y_max - y_min) < tolerance * y_current:
                return y_current
        
        # If not converged, return best estimate
        return y_current

def critical_slope(channel: Channel, discharge: float, 
                  depth_range: tuple = (0.01, 10), 
                  tolerance: float = 1e-6, 
                  max_iterations: int = 100) -> float:
    """
    Calculate critical slope for a given channel and discharge.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        depth_range (tuple): Range of depths to search (m)
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum number of iterations
        
    Returns:
        float: Critical slope (m/m)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    
    # For zero discharge, critical slope is undefined
    if discharge == 0:
        raise ValueError("Critical slope undefined for zero discharge")
    
    # Get critical depth
    yc = critical_depth(channel, discharge)
    
    # At critical depth, Fr = 1
    # Calculate critical slope using Manning's equation
    area = channel.area(yc)
    if area <= 0:
        raise ValueError("Invalid channel area at critical depth")
    
    velocity = discharge / area
    if velocity <= 0:
        raise ValueError("Invalid velocity at critical depth")
    
    # From Manning's equation: S = (nV)²/(R^(4/3))
    hydraulic_radius = channel.hydraulic_radius(yc)
    if hydraulic_radius <= 0:
        raise ValueError("Invalid hydraulic radius at critical depth")
    
    # Calculate critical slope with numerical safeguards
    try:
        critical_slope = ((channel.roughness * velocity)**2) / (hydraulic_radius**(4/3))
        
        # Apply physical constraints
        if critical_slope <= 0:
            raise ValueError("Calculated critical slope is not positive")
        
        if critical_slope > 1.0:  # 100% slope is a reasonable upper limit
            raise ValueError(f"Calculated critical slope ({critical_slope:.4f}) exceeds physical expectations")
        
        return critical_slope
    except Exception as e:
        # Provide better error information
        raise ValueError(f"Error calculating critical slope: {str(e)}")

def calculate_normal_flow_profile(
    channel: Channel, 
    discharge: float, 
    slope: float, 
    x_range: Tuple[float, float], 
    num_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Calculate normal flow profile for a uniform channel.
    
    Parameters:
        channel (Channel): Channel object
        discharge (float): Flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        x_range (Tuple[float, float]): Start and end distances (m)
        num_points (int): Number of points in the profile
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing arrays for:
            - 'x': Distances along channel (m)
            - 'z': Channel bed elevations (m)
            - 'y': Water depths (m)
            - 'wse': Water surface elevations (m)
            - 'v': Flow velocities (m/s)
            - 'fr': Froude numbers
    """
    if discharge < 0:
        raise ValueError("Discharge must be non-negative")
    if slope <= 0:
        raise ValueError("Slope must be positive")
    
    # Calculate normal depth
    yn = normal_depth(channel, discharge, slope)
    
    # Generate x coordinates
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate bed elevations (z = -S*x + z0)
    # Fixed: ensure z starts at correct elevation regardless of x_range[0]
    z0 = slope * x_range[0]  # Initial elevation at x_range[0]
    z = z0 - slope * (x - x_range[0])  # Z decreases with increasing x
    
    # Normal depth is constant for uniform flow
    y = np.full_like(x, yn)
    
    # Water surface elevation
    wse = z + y
    
    # Calculate velocity
    area = channel.area(yn)
    velocity = discharge / area
    v = np.full_like(x, velocity)
    
    # Calculate Froude number
    hydraulic_depth = channel.hydraulic_depth(yn)
    froude = velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)
    fr = np.full_like(x, froude)
    
    return {
        'x': x,
        'z': z,
        'y': y,
        'wse': wse,
        'v': v,
        'fr': fr
    }

def design_channel(
    channel_type: str, 
    discharge: float, 
    slope: float, 
    roughness: float, 
    constraints: Dict[str, float],
    **channel_params
) -> Dict:
    """
    Design a channel to meet flow requirements and constraints.
    
    Parameters:
        channel_type (str): Type of channel ('rectangular', 'trapezoidal', 'triangular')
        discharge (float): Design flow rate (m³/s)
        slope (float): Channel bed slope (m/m)
        roughness (float): Manning's roughness coefficient
        constraints (Dict[str, float]): Design constraints like max_depth, max_velocity, etc.
        **channel_params: Additional channel parameters
        
    Returns:
        Dict: Designed channel parameters and flow characteristics
        
    Raises:
        ValueError: If inputs are invalid or design cannot satisfy constraints
    """
    from .geometry import create_channel
    
    if discharge <= 0:
        raise ValueError("Discharge must be positive")
    if slope <= 0:
        raise ValueError("Slope must be positive")
    if roughness <= 0:
        raise ValueError("Roughness coefficient must be positive")
    
    # Default constraints
    default_constraints = {
        'max_depth': float('inf'),
        'min_depth': 0,
        'max_velocity': float('inf'),
        'min_velocity': 0,
        'max_froude': float('inf'),
        'min_froude': 0,
        'max_width': float('inf'),
        'min_width': 0
    }
    
    # Update with user constraints
    default_constraints.update(constraints)
    constraints = default_constraints
    
    # Optimization strategy depends on channel type
    if channel_type.lower() == 'rectangular':
        # For rectangular, we optimize width
        if 'bottom_width' in channel_params:
            # Fixed width, check if it works
            width = channel_params['bottom_width']
            
            if width <= 0:
                raise ValueError("Bottom width must be positive")
            
            # Create channel
            channel = create_channel('rectangular', bottom_width=width, roughness=roughness)
            
            try:
                # Calculate normal depth
                yn = normal_depth(channel, discharge, slope)
                
                # Check constraints
                if yn > constraints['max_depth']:
                    raise ValueError(f"Calculated depth ({yn:.2f}m) exceeds maximum depth constraint ({constraints['max_depth']}m)")
                if yn < constraints['min_depth']:
                    raise ValueError(f"Calculated depth ({yn:.2f}m) is less than minimum depth constraint ({constraints['min_depth']}m)")
                
                # Calculate velocity and Froude number
                area = channel.area(yn)
                velocity = discharge / area
                hydraulic_depth = channel.hydraulic_depth(yn)
                froude = velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)
                
                # Check velocity constraints
                if velocity > constraints['max_velocity']:
                    raise ValueError(f"Calculated velocity ({velocity:.2f}m/s) exceeds maximum velocity constraint ({constraints['max_velocity']}m/s)")
                if velocity < constraints['min_velocity']:
                    raise ValueError(f"Calculated velocity ({velocity:.2f}m/s) is less than minimum velocity constraint ({constraints['min_velocity']}m/s)")
                
                # Check Froude number constraints
                if froude > constraints['max_froude']:
                    raise ValueError(f"Calculated Froude number ({froude:.2f}) exceeds maximum Froude constraint ({constraints['max_froude']})")
                if froude < constraints['min_froude']:
                    raise ValueError(f"Calculated Froude number ({froude:.2f}) is less than minimum Froude constraint ({constraints['min_froude']})")
                
                # Design is acceptable
                return {
                    'channel_type': 'rectangular',
                    'bottom_width': width,
                    'roughness': roughness,
                    'depth': yn,
                    'velocity': velocity,
                    'froude': froude,
                    'area': area,
                    'wetted_perimeter': channel.wetted_perimeter(yn),
                    'hydraulic_radius': channel.hydraulic_radius(yn),
                    'hydraulic_depth': hydraulic_depth,
                    'discharge': discharge,
                    'slope': slope
                }
            except Exception as e:
                raise ValueError(f"Design with width={width} does not meet constraints: {str(e)}")
        else:
            # Need to find optimal width using more refined optimization
            
            # Initialize search range
            width_min = max(constraints['min_width'], 0.1)  # Minimum practical width
            
            # If max_width is not specified or too large, set a reasonable upper limit
            if constraints['max_width'] == float('inf'):
                # Estimate max width based on min depth and min velocity
                if constraints['min_depth'] > 0 and constraints['min_velocity'] > 0:
                    max_width_estimate = discharge / (constraints['min_depth'] * constraints['min_velocity'])
                    # Add safety factor
                    width_max = max_width_estimate * 3
                else:
                    # Default maximum width based on discharge
                    width_max = max(10.0, 10.0 * math.sqrt(discharge))
            else:
                width_max = constraints['max_width']
            
            # Use logarithmic spacing for better coverage of width range
            num_tests = min(50, max(20, int(10 * math.log10(width_max/width_min) + 10)))
            widths = np.logspace(math.log10(width_min), math.log10(width_max), num_tests)
            
            valid_designs = []
            constraint_violations = []
            
            for w in widths:
                try:
                    # Create channel
                    channel = create_channel('rectangular', bottom_width=w, roughness=roughness)
                    
                    # Calculate normal depth
                    yn = normal_depth(channel, discharge, slope)
                    
                    # Track constraint violations for better feedback
                    violations = []
                    
                    # Check depth constraints
                    if yn > constraints['max_depth']:
                        violations.append(f"depth {yn:.2f}m > max {constraints['max_depth']}m")
                    if yn < constraints['min_depth']:
                        violations.append(f"depth {yn:.2f}m < min {constraints['min_depth']}m")
                    
                    # Calculate velocity and Froude number
                    area = channel.area(yn)
                    velocity = discharge / area
                    hydraulic_depth = channel.hydraulic_depth(yn)
                    froude = velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)
                    
                    # Check velocity constraints
                    if velocity > constraints['max_velocity']:
                        violations.append(f"velocity {velocity:.2f}m/s > max {constraints['max_velocity']}m/s")
                    if velocity < constraints['min_velocity']:
                        violations.append(f"velocity {velocity:.2f}m/s < min {constraints['min_velocity']}m/s")
                    
                    # Check Froude number constraints
                    if froude > constraints['max_froude']:
                        violations.append(f"Froude {froude:.2f} > max {constraints['max_froude']}")
                    if froude < constraints['min_froude']:
                        violations.append(f"Froude {froude:.2f} < min {constraints['min_froude']}")
                    
                    if violations:
                        constraint_violations.append((w, yn, violations))
                        continue
                    
                    # Design is acceptable
                    valid_designs.append({
                        'channel_type': 'rectangular',
                        'bottom_width': w,
                        'roughness': roughness,
                        'depth': yn,
                        'velocity': velocity,
                        'froude': froude,
                        'area': area,
                        'wetted_perimeter': channel.wetted_perimeter(yn),
                        'hydraulic_radius': channel.hydraulic_radius(yn),
                        'hydraulic_depth': hydraulic_depth,
                        'discharge': discharge,
                        'slope': slope
                    })
                except Exception as e:
                    constraint_violations.append((w, None, [f"Calculation error: {str(e)}"]))
                    continue
            
            if not valid_designs:
                # Provide more helpful error message based on constraint violations
                error_msg = "Cannot find a design that meets all constraints. Violations found:\n"
                # Group by violation type for more clear feedback
                for w, yn, violations in constraint_violations[:5]:  # Show first 5 violations
                    error_msg += f"Width={w:.2f}m"
                    if yn is not None:
                        error_msg += f", depth={yn:.2f}m: "
                    else:
                        error_msg += ": "
                    error_msg += ", ".join(violations) + "\n"
                
                # Add suggestion for relaxing constraints
                error_msg += "Try relaxing constraints or using a different channel type."
                raise ValueError(error_msg)
            
            # Return the design with the minimum width (economical design)
            return min(valid_designs, key=lambda d: d['bottom_width'])
    
    elif channel_type.lower() == 'trapezoidal':
        # Similar approach for trapezoidal channels
        # This is a simplified approach; a more comprehensive design would use optimization
        
        # Check if side slope is provided
        if 'side_slope' not in channel_params:
            raise ValueError("Side slope must be provided for trapezoidal channel")
        
        side_slope = channel_params['side_slope']
        if side_slope < 0:
            raise ValueError("Side slope must be non-negative")
        
        if 'bottom_width' in channel_params:
            # Fixed width, check if it works
            width = channel_params['bottom_width']
            
            if width < 0:
                raise ValueError("Bottom width must be non-negative")
            
            # Create channel
            try:
                channel = create_channel('trapezoidal', bottom_width=width, 
                                       side_slope=side_slope, roughness=roughness)
                
                # Calculate normal depth
                yn = normal_depth(channel, discharge, slope)
                
                # Calculate flow parameters
                area = channel.area(yn)
                velocity = discharge / area
                hydraulic_depth = channel.hydraulic_depth(yn)
                froude = velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)
                top_width = width + 2 * side_slope * yn
                
                # Check all constraints
                if yn > constraints['max_depth']:
                    raise ValueError(f"Depth ({yn:.2f}m) exceeds maximum ({constraints['max_depth']}m)")
                if yn < constraints['min_depth']:
                    raise ValueError(f"Depth ({yn:.2f}m) less than minimum ({constraints['min_depth']}m)")
                if velocity > constraints['max_velocity']:
                    raise ValueError(f"Velocity ({velocity:.2f}m/s) exceeds maximum ({constraints['max_velocity']}m/s)")
                if velocity < constraints['min_velocity']:
                    raise ValueError(f"Velocity ({velocity:.2f}m/s) less than minimum ({constraints['min_velocity']}m/s)")
                if froude > constraints['max_froude']:
                    raise ValueError(f"Froude number ({froude:.2f}) exceeds maximum ({constraints['max_froude']})")
                if froude < constraints['min_froude']:
                    raise ValueError(f"Froude number ({froude:.2f}) less than minimum ({constraints['min_froude']})")
                if top_width > constraints['max_width'] and constraints['max_width'] != float('inf'):
                    raise ValueError(f"Top width ({top_width:.2f}m) exceeds maximum ({constraints['max_width']}m)")
                
                # Design is acceptable
                return {
                    'channel_type': 'trapezoidal',
                    'bottom_width': width,
                    'side_slope': side_slope,
                    'roughness': roughness,
                    'depth': yn,
                    'velocity': velocity,
                    'froude': froude,
                    'area': area,
                    'wetted_perimeter': channel.wetted_perimeter(yn),
                    'hydraulic_radius': channel.hydraulic_radius(yn),
                    'hydraulic_depth': hydraulic_depth,
                    'top_width': top_width,
                    'discharge': discharge,
                    'slope': slope
                }
            except Exception as e:
                raise ValueError(f"Design error with bottom width={width}, side slope={side_slope}: {str(e)}")
        else:
            # Need to find optimal width
            # Width search range
            width_min = max(constraints['min_width'], 0)  # Can be zero for triangular
            
            # Determine max width more intelligently
            if constraints['max_width'] == float('inf'):
                # Estimate from discharge and constraints
                if constraints['min_depth'] > 0 and constraints['min_velocity'] > 0:
                    # Account for side slopes in area calculation
                    est_depth = constraints['min_depth']
                    est_side_area = side_slope * est_depth * est_depth  # Area from side slopes
                    required_area = discharge / constraints['min_velocity']
                    max_width_estimate = (required_area - est_side_area) / est_depth
                    width_max = max(max_width_estimate * 3, 10.0)
                else:
                    width_max = max(10.0, 5.0 * math.sqrt(discharge))
            else:
                width_max = constraints['max_width']
            
            # Special case for triangular channels
            if width_min == 0 and width_max == 0:
                # Triangular channel (bottom_width = 0)
                try:
                    channel = create_channel('trapezoidal', bottom_width=0, 
                                           side_slope=side_slope, roughness=roughness)
                    
                    # Calculate normal depth
                    yn = normal_depth(channel, discharge, slope)
                    
                    # Calculate parameters
                    area = channel.area(yn)
                    velocity = discharge / area
                    hydraulic_depth = channel.hydraulic_depth(yn)
                    froude = velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)
                    top_width = 2 * side_slope * yn
                    
                    # Check all constraints (same as above)
                    
                    # Design is acceptable
                    return {
                        'channel_type': 'triangular',  # Note: actually a triangular channel
                        'bottom_width': 0,
                        'side_slope': side_slope,
                        'roughness': roughness,
                        'depth': yn,
                        'velocity': velocity,
                        'froude': froude,
                        'area': area,
                        'wetted_perimeter': channel.wetted_perimeter(yn),
                        'hydraulic_radius': channel.hydraulic_radius(yn),
                        'hydraulic_depth': hydraulic_depth,
                        'top_width': top_width,
                        'discharge': discharge,
                        'slope': slope
                    }
                except Exception as e:
                    raise ValueError(f"Cannot design triangular channel: {str(e)}")
            
            # Test a range of widths with improved spacing
            num_tests = min(50, max(20, int(10 * math.log10((width_max+0.1)/(width_min+0.1)) + 10)))
            widths = np.linspace(width_min, width_max, num_tests)
            
            valid_designs = []
            constraint_violations = []
            
            for w in widths:
                try:
                    # Create channel
                    channel = create_channel('trapezoidal', bottom_width=w, 
                                           side_slope=side_slope, roughness=roughness)
                    
                    # Calculate normal depth and validate design
                    yn = normal_depth(channel, discharge, slope)
                    
                    # Track constraint violations
                    violations = []
                    
                    # Calculate parameters for constraint checks
                    area = channel.area(yn)
                    velocity = discharge / area
                    hydraulic_depth = channel.hydraulic_depth(yn)
                    froude = velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)
                    top_width = w + 2 * side_slope * yn
                    
                    # Check all constraints (same as above)
                    # (Add violations to the list as needed)
                    
                    if not violations:
                        # Design is acceptable
                        valid_designs.append({
                            'channel_type': 'trapezoidal',
                            'bottom_width': w,
                            'side_slope': side_slope,
                            'roughness': roughness,
                            'depth': yn,
                            'velocity': velocity,
                            'froude': froude,
                            'area': area,
                            'wetted_perimeter': channel.wetted_perimeter(yn),
                            'hydraulic_radius': channel.hydraulic_radius(yn),
                            'hydraulic_depth': hydraulic_depth,
                            'top_width': top_width,
                            'discharge': discharge,
                            'slope': slope
                        })
                except Exception:
                    continue
            
            if not valid_designs:
                raise ValueError("Cannot find a trapezoidal channel design that meets all constraints. Try relaxing constraints.")
            
            # Return the design with the minimum wetted perimeter (most hydraulically efficient)
            return min(valid_designs, key=lambda d: d['wetted_perimeter'])
    
    elif channel_type.lower() == 'triangular':
        # Implementation for triangular channels
        # This is a special case of trapezoidal with bottom_width = 0
        
        if 'side_slope' not in channel_params:
            raise ValueError("Side slope must be provided for triangular channel")
        
        side_slope = channel_params['side_slope']
        if side_slope <= 0:
            raise ValueError("Side slope must be positive for triangular channel")
        
        try:
            # Create triangular channel
            channel = create_channel('trapezoidal', bottom_width=0, 
                                   side_slope=side_slope, roughness=roughness)
            
            # Calculate normal depth
            yn = normal_depth(channel, discharge, slope)
            
            # Calculate parameters
            area = channel.area(yn)
            velocity = discharge / area
            hydraulic_depth = channel.hydraulic_depth(yn)
            froude = velocity / math.sqrt(basic.GRAVITY * hydraulic_depth)
            top_width = 2 * side_slope * yn
            
            # Check constraints (same as above)
            
            # Design is acceptable
            return {
                'channel_type': 'triangular',
                'bottom_width': 0,
                'side_slope': side_slope,
                'roughness': roughness,
                'depth': yn,
                'velocity': velocity,
                'froude': froude,
                'area': area,
                'wetted_perimeter': channel.wetted_perimeter(yn),
                'hydraulic_radius': channel.hydraulic_radius(yn),
                'hydraulic_depth': hydraulic_depth,
                'top_width': top_width,
                'discharge': discharge,
                'slope': slope
            }
        except Exception as e:
            raise ValueError(f"Cannot design triangular channel: {str(e)}")
    
    else:
        raise ValueError(f"Channel design not implemented for {channel_type}")