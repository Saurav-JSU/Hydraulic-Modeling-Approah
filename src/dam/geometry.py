"""
Dam geometry definitions and calculations.

This module provides classes and functions for defining different dam
types and calculating their geometric properties.
"""

import math
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np

class Dam:
    """Base class for all dam types."""
    
    def __init__(self, height: float, crest_elevation: float):
        """
        Initialize a dam with height and crest elevation.
        
        Parameters:
            height (float): Dam height (m)
            crest_elevation (float): Elevation of dam crest (m)
        """
        if height <= 0:
            raise ValueError("Dam height must be positive")
        
        self.height = height
        self.crest_elevation = crest_elevation
        self.base_elevation = crest_elevation - height
    
    def get_profile(self, stations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get the geometric profile of the dam.
        
        Parameters:
            stations (np.ndarray): Array of x-coordinates for profile
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with x and z coordinates
        """
        raise NotImplementedError("Subclasses must implement get_profile()")
    
    def get_discharge_coefficient(self, head: float) -> float:
        """
        Get the discharge coefficient for flow over the dam.
        
        Parameters:
            head (float): Head above crest (m)
            
        Returns:
            float: Discharge coefficient
        """
        raise NotImplementedError("Subclasses must implement get_discharge_coefficient()")
    
    def get_overflow_area(self, water_elevation: float, width: float) -> float:
        """
        Calculate the overflow area for a given water elevation.
        
        Parameters:
            water_elevation (float): Water surface elevation (m)
            width (float): Width of dam crest (m)
            
        Returns:
            float: Overflow area (m²)
        """
        if water_elevation <= self.crest_elevation:
            return 0
        
        # For a simple dam, overflow area is rectangular
        overflow_height = water_elevation - self.crest_elevation
        return overflow_height * width
    
    def get_upstream_water_elevation(self, discharge: float, width: float) -> float:
        """
        Calculate the upstream water elevation for a given discharge.
        
        Parameters:
            discharge (float): Flow rate (m³/s)
            width (float): Width of dam crest (m)
            
        Returns:
            float: Upstream water elevation (m)
            
        Note:
            This is an inverse calculation from discharge to head,
            often requiring an iterative solution.
        """
        raise NotImplementedError("Subclasses must implement get_upstream_water_elevation()")


class BroadCrestedWeir(Dam):
    """Broad-crested weir dam."""
    
    def __init__(self, height: float, crest_elevation: float, crest_width: float):
        """
        Initialize a broad-crested weir.
        
        Parameters:
            height (float): Dam height (m)
            crest_elevation (float): Elevation of dam crest (m)
            crest_width (float): Width of dam crest in flow direction (m)
        """
        super().__init__(height, crest_elevation)
        
        if crest_width <= 0:
            raise ValueError("Crest width must be positive")
        
        self.crest_width = crest_width
    
    def get_profile(self, stations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get the geometric profile of the broad-crested weir.
        
        Parameters:
            stations (np.ndarray): Array of x-coordinates for profile
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with x and z coordinates
        """
        # Create a simple rectangular profile
        elevations = np.zeros_like(stations)
        
        # Upstream face (vertical)
        upstream_face = stations <= 0
        elevations[upstream_face] = self.base_elevation
        
        # Crest (horizontal)
        crest = (stations > 0) & (stations <= self.crest_width)
        elevations[crest] = self.crest_elevation
        
        # Downstream face (vertical)
        downstream_face = stations > self.crest_width
        elevations[downstream_face] = self.base_elevation
        
        return {
            'x': stations,
            'z': elevations
        }
    
    def get_discharge_coefficient(self, head: float) -> float:
        """
        Get the discharge coefficient for a broad-crested weir.
        
        Parameters:
            head (float): Head above crest (m)
            
        Returns:
            float: Discharge coefficient
        
        Note:
            Improved coefficients based on hydraulic lab data for broad-crested weirs.
            Reflects the transition between modular and non-modular flow regimes.
        """
        if head <= 0:
            return 0
        
        # Calculate head to crest width ratio
        h_l_ratio = head / self.crest_width
        
        # Improved coefficient calculation based on Hager & Schwalt (1994)
        # and Robust experimental data from USBR and USACE
        if h_l_ratio <= 0.08:
            # Very low heads - approaching zero discharge
            return max(0.28, 0.33 - 0.60 * h_l_ratio)
        elif h_l_ratio <= 0.33:
            # Normal operating range for broad-crested weirs
            return 0.33 + 0.18 * (h_l_ratio - 0.08) / 0.25
        else:
            # High head to crest length ratio - approaching sharp-crested behavior
            return min(0.385, 0.51 - 0.33 * h_l_ratio)
    
    def calculate_discharge(self, head: float, width: float) -> float:
        """
        Calculate discharge over a broad-crested weir.
        
        Parameters:
            head (float): Head above crest (m)
            width (float): Width of dam perpendicular to flow (m)
            
        Returns:
            float: Discharge (m³/s)
        """
        if head <= 0:
            return 0
        
        # Get discharge coefficient
        Cd = self.get_discharge_coefficient(head)
        
        # Standard weir equation: Q = Cd * L * H^(3/2) * sqrt(2g)
        g = 9.81  # Gravitational acceleration (m/s²)
        return Cd * width * head**(3/2) * math.sqrt(2 * g)
    
    def get_upstream_water_elevation(self, discharge: float, width: float, 
                                   tolerance: float = 1e-6, 
                                   max_iterations: int = 100) -> float:
        """
        Calculate the upstream water elevation for a given discharge.
        
        Parameters:
            discharge (float): Flow rate (m³/s)
            width (float): Width of dam perpendicular to flow (m)
            tolerance (float, optional): Convergence tolerance
            max_iterations (int, optional): Maximum number of iterations
            
        Returns:
            float: Upstream water elevation (m)
        """
        if discharge <= 0:
            return self.crest_elevation
        
        # Initial guess for head - improved based on typical broad-crested weir behavior
        # More accurate initial estimate reduces iterations needed
        g = 9.81  # Gravitational acceleration (m/s²)
        head_guess = (discharge / (0.33 * width * math.sqrt(2 * g)))**(2/3)
        
        # Bounds for physically realistic solutions
        min_head = 0.001  # Minimum practical head value (m)
        max_head = 10.0 * head_guess  # Maximum reasonable head
        
        # Adaptive damping parameters
        initial_damping = 0.8
        min_damping = 0.2
        
        # Iterative solution with adaptive damping for robust convergence
        damping = initial_damping
        prev_error = float('inf')
        
        for i in range(max_iterations):
            # Calculate discharge for current head guess
            calculated_discharge = self.calculate_discharge(head_guess, width)
            current_error = abs(calculated_discharge - discharge)
            
            # Check convergence
            if current_error < tolerance:
                break
            
            # Adjust damping if oscillating or slow convergence
            if i > 3 and current_error > 0.8 * prev_error:
                # Reduce damping factor if not converging well
                damping = max(damping * 0.7, min_damping)
            
            # Calculate derivative for Newton-Raphson method
            # Use central difference for better numerical stability
            delta_h = max(0.001 * head_guess, 0.0001)
            disch_plus = self.calculate_discharge(head_guess + delta_h, width)
            disch_minus = self.calculate_discharge(max(head_guess - delta_h, 0.0001), width)
            dQ_dH = (disch_plus - disch_minus) / (2 * delta_h)
            
            # Apply safe bounds to derivative to prevent divide-by-zero or instability
            if abs(dQ_dH) < 1e-8:
                dQ_dH = 1e-8 if dQ_dH >= 0 else -1e-8
            
            # Update head with damped Newton-Raphson
            correction = (calculated_discharge - discharge) / dQ_dH
            head_guess -= damping * correction
            
            # Ensure head stays within physical bounds
            head_guess = max(min_head, min(max_head, head_guess))
            
            # Store error for next iteration comparison
            prev_error = current_error
        
        # Return water surface elevation
        return self.crest_elevation + head_guess


class SharpCrestedWeir(Dam):
    """Sharp-crested weir dam."""
    
    def get_profile(self, stations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get the geometric profile of the sharp-crested weir.
        
        Parameters:
            stations (np.ndarray): Array of x-coordinates for profile
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with x and z coordinates
        """
        # Create a simple triangular profile
        elevations = np.zeros_like(stations)
        
        # Upstream face (vertical)
        upstream_face = stations <= 0
        elevations[upstream_face] = self.base_elevation
        
        # Crest (point)
        crest = np.abs(stations) < 0.01
        elevations[crest] = self.crest_elevation
        
        # Downstream face (steep slope)
        downstream_face = stations > 0
        elevations[downstream_face] = np.maximum(
            self.crest_elevation - 5 * stations[downstream_face],
            self.base_elevation
        )
        
        return {
            'x': stations,
            'z': elevations
        }
    
    def get_discharge_coefficient(self, head: float) -> float:
        """
        Get the discharge coefficient for a sharp-crested weir.
        
        Parameters:
            head (float): Head above crest (m)
            
        Returns:
            float: Discharge coefficient
            
        Note:
            Improved coefficients based on Rehbock formula and USBR research,
            considering approach velocity and nappe ventilation effects.
        """
        if head <= 0:
            return 0
        
        # Calculate head to height ratio (H/P)
        h_p_ratio = head / self.height
        
        # Improved coefficient calculation based on Rehbock formula
        # and USBR/USACE research for ventilated sharp-crested weirs
        base_coefficient = 0.611 + 0.075 * (head / (head + self.height))
        
        # Adjustment for very small H/P ratios (approaching zero flow)
        if h_p_ratio < 0.03:
            return max(0.55, base_coefficient - 0.1)
        
        # Adjustment for approach velocity and nappe effects
        if h_p_ratio <= 0.3:
            # Lower H/P ratios: fully ventilated nappe
            return base_coefficient
        elif h_p_ratio <= 1.0:
            # Mid-range H/P ratios: transitional behavior
            # Linear increase to account for approach velocity effects
            return base_coefficient + 0.045 * (h_p_ratio - 0.3) / 0.7
        else:
            # High H/P ratios: significant velocity head contribution
            # Coefficient approaches that of a submerged orifice
            return min(0.7, base_coefficient + 0.045 + 0.01 * (h_p_ratio - 1.0))
    
    def calculate_discharge(self, head: float, width: float) -> float:
        """
        Calculate discharge over a sharp-crested weir.
        
        Parameters:
            head (float): Head above crest (m)
            width (float): Width of dam perpendicular to flow (m)
            
        Returns:
            float: Discharge (m³/s)
        """
        if head <= 0:
            return 0
        
        # Get discharge coefficient
        Cd = self.get_discharge_coefficient(head)
        
        # Standard weir equation: Q = Cd * L * H^(3/2) * sqrt(2g)
        g = 9.81  # Gravitational acceleration (m/s²)
        return Cd * width * head**(3/2) * math.sqrt(2 * g)
    
    def get_upstream_water_elevation(self, discharge: float, width: float, 
                                   tolerance: float = 1e-6, 
                                   max_iterations: int = 100) -> float:
        """
        Calculate the upstream water elevation for a given discharge.
        
        Parameters:
            discharge (float): Flow rate (m³/s)
            width (float): Width of dam perpendicular to flow (m)
            tolerance (float, optional): Convergence tolerance
            max_iterations (int, optional): Maximum number of iterations
            
        Returns:
            float: Upstream water elevation (m)
        """
        if discharge <= 0:
            return self.crest_elevation
        
        # Initial guess for head - improved based on typical sharp-crested weir coefficient
        g = 9.81  # Gravitational acceleration (m/s²)
        head_guess = (discharge / (0.61 * width * math.sqrt(2 * g)))**(2/3)
        
        # Bounds for physically realistic solutions
        min_head = 0.001  # Minimum practical head value (m)
        max_head = 10.0 * head_guess  # Maximum reasonable head
        
        # Adaptive damping parameters
        initial_damping = 0.8
        min_damping = 0.2
        
        # Iterative solution with adaptive damping for robust convergence
        damping = initial_damping
        prev_error = float('inf')
        
        for i in range(max_iterations):
            # Calculate discharge for current head guess
            calculated_discharge = self.calculate_discharge(head_guess, width)
            current_error = abs(calculated_discharge - discharge)
            
            # Check convergence
            if current_error < tolerance:
                break
            
            # Adjust damping if oscillating or slow convergence
            if i > 3 and current_error > 0.8 * prev_error:
                # Reduce damping factor if not converging well
                damping = max(damping * 0.7, min_damping)
            
            # Calculate derivative for Newton-Raphson method
            # Use central difference for better numerical stability
            delta_h = max(0.001 * head_guess, 0.0001)
            disch_plus = self.calculate_discharge(head_guess + delta_h, width)
            disch_minus = self.calculate_discharge(max(head_guess - delta_h, 0.0001), width)
            dQ_dH = (disch_plus - disch_minus) / (2 * delta_h)
            
            # Apply safe bounds to derivative to prevent divide-by-zero or instability
            if abs(dQ_dH) < 1e-8:
                dQ_dH = 1e-8 if dQ_dH >= 0 else -1e-8
            
            # Update head with damped Newton-Raphson
            correction = (calculated_discharge - discharge) / dQ_dH
            head_guess -= damping * correction
            
            # Ensure head stays within physical bounds
            head_guess = max(min_head, min(max_head, head_guess))
            
            # Store error for next iteration comparison
            prev_error = current_error
        
        # Return water surface elevation
        return self.crest_elevation + head_guess


class OgeeWeir(Dam):
    """Ogee spillway dam."""
    
    def __init__(self, height: float, crest_elevation: float, design_head: float, k_value: float = 2.0):
        """
        Initialize an ogee spillway.
        
        Parameters:
            height (float): Dam height (m)
            crest_elevation (float): Elevation of dam crest (m)
            design_head (float): Design head for spillway profile (m)
            k_value (float, optional): Profile constant for ogee shape, default is 2.0
        """
        super().__init__(height, crest_elevation)
        
        if design_head <= 0:
            raise ValueError("Design head must be positive")
        
        self.design_head = design_head
        self.k_value = k_value  # Added parameter for profile customization
    
    def get_profile(self, stations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get the geometric profile of the ogee spillway.
        
        Parameters:
            stations (np.ndarray): Array of x-coordinates for profile
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with x and z coordinates
        """
        # Create an ogee profile
        elevations = np.zeros_like(stations)
        
        # Upstream face - using elliptical approach for hydraulic efficiency
        upstream_face = stations <= 0
        # Create a more hydraulically efficient upstream face with elliptical curve
        us_stations = np.abs(stations[upstream_face])
        # Parameters for upstream elliptical curve: x²/a² + y²/b² = 1
        a = 0.5 * self.height
        b = 0.27 * self.height
        mask = us_stations <= a
        elliptical_drop = np.zeros_like(us_stations)
        elliptical_drop[mask] = b * np.sqrt(1 - (us_stations[mask]/a)**2)
        
        elevations[upstream_face] = np.maximum(
            self.crest_elevation - (self.height - elliptical_drop),
            self.base_elevation
        )
        
        # Crest region - slight rounding for smoother transition
        crest_region = np.abs(stations) < 0.1 * self.design_head
        elevations[crest_region] = self.crest_elevation
        
        # Downstream ogee curve - using WES profile: X^1.85 = K * H^0.85 * Y
        # The k_value parameter allows for different profile shapes
        downstream_face = stations > 0
        
        # Apply ogee equation to downstream face
        # z = crest_elevation - (x^1.85 / (K * H^0.85))
        x_ogee = stations[downstream_face]
        y_ogee = (x_ogee**1.85) / (self.k_value * self.design_head**0.85)
        
        elevations[downstream_face] = np.maximum(
            self.crest_elevation - y_ogee,
            self.base_elevation
        )
        
        return {
            'x': stations,
            'z': elevations
        }
    
    def get_discharge_coefficient(self, head: float) -> float:
        """
        Get the discharge coefficient for an ogee spillway.
        
        Parameters:
            head (float): Head above crest (m)
            
        Returns:
            float: Discharge coefficient
            
        Note:
            Improved coefficients based on USACE hydraulic design criteria
            and comprehensive USBR research for ogee spillways.
        """
        if head <= 0:
            return 0
        
        # Head ratio (actual/design)
        head_ratio = head / self.design_head
        
        # Improved coefficient calculation based on USACE and USBR research
        # with more physically accurate transitions between flow regimes
        if head_ratio < 0.2:
            # Very low flow regime - coefficient increases with head
            return 0.65 + 0.25 * (head_ratio / 0.2)
        elif head_ratio < 0.8:
            # Approaching design flow - smooth transition to optimal coefficient
            return 0.70 + 0.10 * ((head_ratio - 0.2) / 0.6)
        elif head_ratio < 1.0:
            # Near design flow - optimal hydraulic efficiency
            return 0.80 - 0.05 * ((head_ratio - 0.8) / 0.2)
        elif head_ratio < 1.5:
            # Above design flow - slight efficiency increase due to approach velocity
            # but offset by non-optimal crest geometry for this flow level
            base = 0.75 + (0.08 * (head_ratio - 1.0))
            
            # Apply pressure adjustment for higher heads
            # Higher heads create greater negative pressure on spillway face
            pressure_factor = 1.0 + 0.015 * (head_ratio - 1.0)
            
            return base * pressure_factor
        else:
            # Well above design flow - approaching orifice flow behavior
            # with complex pressure distribution on spillway face
            return min(0.87, 0.83 + 0.04 * math.sqrt(head_ratio - 1.5))
    
    def calculate_discharge(self, head: float, width: float) -> float:
        """
        Calculate discharge over an ogee spillway.
        
        Parameters:
            head (float): Head above crest (m)
            width (float): Width of dam perpendicular to flow (m)
            
        Returns:
            float: Discharge (m³/s)
        """
        if head <= 0:
            return 0
        
        # Get discharge coefficient
        Cd = self.get_discharge_coefficient(head)
        
        # Standard weir equation: Q = Cd * L * H^(3/2) * sqrt(2g)
        g = 9.81  # Gravitational acceleration (m/s²)
        return Cd * width * head**(3/2) * math.sqrt(2 * g)
    
    def get_upstream_water_elevation(self, discharge: float, width: float, 
                                   tolerance: float = 1e-6, 
                                   max_iterations: int = 100) -> float:
        """
        Calculate the upstream water elevation for a given discharge.
        
        Parameters:
            discharge (float): Flow rate (m³/s)
            width (float): Width of dam perpendicular to flow (m)
            tolerance (float, optional): Convergence tolerance
            max_iterations (int, optional): Maximum number of iterations
            
        Returns:
            float: Upstream water elevation (m)
        """
        if discharge <= 0:
            return self.crest_elevation
        
        # Initial guess for head - improved based on typical design conditions
        # and discharge coefficient at design head
        g = 9.81  # Gravitational acceleration (m/s²)
        
        # Use different coefficient estimates based on relative discharge magnitude
        # to improve initial guess accuracy
        rel_discharge = discharge / (width * self.design_head**(3/2) * math.sqrt(2 * g))
        
        if rel_discharge < 0.5:
            # Low flow condition
            head_guess = (discharge / (0.68 * width * math.sqrt(2 * g)))**(2/3)
        elif rel_discharge < 1.0:
            # Mid-range flow condition
            head_guess = (discharge / (0.75 * width * math.sqrt(2 * g)))**(2/3)
        else:
            # High flow condition - use design head as reference
            head_guess = self.design_head * (rel_discharge / 0.75)**(2/3)
        
        # Bounds for physically realistic solutions
        min_head = 0.001  # Minimum practical head value (m)
        max_head = max(10.0 * head_guess, 3.0 * self.design_head)  # Maximum reasonable head
        
        # Adaptive damping parameters
        initial_damping = 0.8
        min_damping = 0.2
        
        # Iterative solution with adaptive damping for robust convergence
        damping = initial_damping
        prev_error = float('inf')
        
        for i in range(max_iterations):
            # Calculate discharge for current head guess
            calculated_discharge = self.calculate_discharge(head_guess, width)
            current_error = abs(calculated_discharge - discharge)
            
            # Check convergence
            if current_error < tolerance:
                break
            
            # Adjust damping if oscillating or slow convergence
            if i > 3 and current_error > 0.8 * prev_error:
                # Reduce damping factor if not converging well
                damping = max(damping * 0.7, min_damping)
            
            # Calculate derivative for Newton-Raphson method
            # Use central difference for better numerical stability
            delta_h = max(0.001 * head_guess, 0.0001)
            disch_plus = self.calculate_discharge(head_guess + delta_h, width)
            disch_minus = self.calculate_discharge(max(head_guess - delta_h, 0.0001), width)
            dQ_dH = (disch_plus - disch_minus) / (2 * delta_h)
            
            # Apply safe bounds to derivative to prevent divide-by-zero or instability
            if abs(dQ_dH) < 1e-8:
                dQ_dH = 1e-8 if dQ_dH >= 0 else -1e-8
            
            # Update head with damped Newton-Raphson
            correction = (calculated_discharge - discharge) / dQ_dH
            head_guess -= damping * correction
            
            # Ensure head stays within physical bounds
            head_guess = max(min_head, min(max_head, head_guess))
            
            # Store error for next iteration comparison
            prev_error = current_error
        
        # Return water surface elevation
        return self.crest_elevation + head_guess


def create_dam(dam_type: str, **kwargs) -> Dam:
    """
    Factory function to create dam objects.
    
    Parameters:
        dam_type (str): Type of dam ('broad_crested', 'sharp_crested', 'ogee')
        **kwargs: Dam parameters
        
    Returns:
        Dam: Dam object
        
    Raises:
        ValueError: If dam type is unknown
    """
    dam_type = dam_type.lower()
    
    if dam_type == 'broad_crested':
        required = {'height', 'crest_elevation', 'crest_width'}
        if not required.issubset(kwargs.keys()):
            missing = required - set(kwargs.keys())
            raise ValueError(f"Missing parameters for broad-crested weir: {missing}")
        
        return BroadCrestedWeir(kwargs['height'], kwargs['crest_elevation'], kwargs['crest_width'])
    
    elif dam_type == 'sharp_crested':
        required = {'height', 'crest_elevation'}
        if not required.issubset(kwargs.keys()):
            missing = required - set(kwargs.keys())
            raise ValueError(f"Missing parameters for sharp-crested weir: {missing}")
        
        return SharpCrestedWeir(kwargs['height'], kwargs['crest_elevation'])
    
    elif dam_type == 'ogee':
        required = {'height', 'crest_elevation', 'design_head'}
        if not required.issubset(kwargs.keys()):
            missing = required - set(kwargs.keys())
            raise ValueError(f"Missing parameters for ogee spillway: {missing}")
        
        # Handle optional k_value parameter
        k_value = kwargs.get('k_value', 2.0)
        
        return OgeeWeir(kwargs['height'], kwargs['crest_elevation'], kwargs['design_head'], k_value)
    
    else:
        raise ValueError(f"Unknown dam type: {dam_type}")