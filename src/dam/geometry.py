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
        """
        if head <= 0:
            return 0
        
        # Basic coefficient for broad-crested weir
        # Cd typically ranges from 0.33 to 0.36
        base_coefficient = 0.35
        
        # Adjust coefficient based on H/L ratio (head to crest width)
        h_l_ratio = head / self.crest_width
        
        if h_l_ratio <= 0.1:
            # For very low heads relative to crest width
            return 0.33
        elif h_l_ratio >= 0.4:
            # For higher heads relative to crest width
            return 0.36
        else:
            # Linear interpolation for intermediate values
            return 0.33 + (0.36 - 0.33) * (h_l_ratio - 0.1) / 0.3
    
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
        
        # Q = Cd * L * H^(3/2) * sqrt(2g)
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
        
        # Initial guess for head
        head_guess = (discharge / (0.35 * width * math.sqrt(2 * 9.81)))**(2/3)
        
        # Iterative solution
        for i in range(max_iterations):
            # Calculate discharge for current head guess
            calculated_discharge = self.calculate_discharge(head_guess, width)
            
            # Check convergence
            if abs(calculated_discharge - discharge) < tolerance:
                break
            
            # Adjust head guess
            # Use a damped Newton-Raphson approach
            # dQ/dH ≈ 1.5 * Cd * L * H^(0.5) * sqrt(2g)
            h_pow = head_guess**(0.5)
            if h_pow < 1e-10:
                # Avoid division by zero
                h_pow = 1e-10
                
            dQ_dH = 1.5 * self.get_discharge_coefficient(head_guess) * width * h_pow * math.sqrt(2 * 9.81)
            
            # Damping factor
            damping = 0.7
            
            # Update head
            correction = (calculated_discharge - discharge) / dQ_dH
            head_guess -= damping * correction
            
            # Ensure head is positive
            if head_guess <= 0:
                head_guess = 0.01
        
        # Return elevation
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
        """
        if head <= 0:
            return 0
        
        # Basic coefficient for sharp-crested weir
        # Cd typically around 0.62
        base_coefficient = 0.62
        
        # Adjust coefficient based on H/P ratio (head to height)
        h_p_ratio = head / self.height
        
        if h_p_ratio <= 0.1:
            # For very low heads relative to weir height
            return 0.60
        elif h_p_ratio >= 0.4:
            # For higher heads relative to weir height
            return 0.64
        else:
            # Linear interpolation for intermediate values
            return 0.60 + (0.64 - 0.60) * (h_p_ratio - 0.1) / 0.3
    
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
        
        # Q = Cd * L * H^(3/2) * sqrt(2g)
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
        
        # Initial guess for head
        head_guess = (discharge / (0.62 * width * math.sqrt(2 * 9.81)))**(2/3)
        
        # Iterative solution (similar to broad-crested weir)
        for i in range(max_iterations):
            calculated_discharge = self.calculate_discharge(head_guess, width)
            
            if abs(calculated_discharge - discharge) < tolerance:
                break
            
            # Adjust head guess
            h_pow = head_guess**(0.5)
            if h_pow < 1e-10:
                h_pow = 1e-10
                
            dQ_dH = 1.5 * self.get_discharge_coefficient(head_guess) * width * h_pow * math.sqrt(2 * 9.81)
            damping = 0.7
            correction = (calculated_discharge - discharge) / dQ_dH
            head_guess -= damping * correction
            
            if head_guess <= 0:
                head_guess = 0.01
        
        return self.crest_elevation + head_guess


class OgeeWeir(Dam):
    """Ogee spillway dam."""
    
    def __init__(self, height: float, crest_elevation: float, design_head: float):
        """
        Initialize an ogee spillway.
        
        Parameters:
            height (float): Dam height (m)
            crest_elevation (float): Elevation of dam crest (m)
            design_head (float): Design head for spillway profile (m)
        """
        super().__init__(height, crest_elevation)
        
        if design_head <= 0:
            raise ValueError("Design head must be positive")
        
        self.design_head = design_head
    
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
        
        # Upstream face (vertical)
        upstream_face = stations <= 0
        elevations[upstream_face] = np.maximum(
            self.crest_elevation - 0.5 * np.abs(stations[upstream_face]),
            self.base_elevation
        )
        
        # Crest and downstream ogee curve
        # Using WES profile: X^1.85 = K * H^0.85 * Y
        k = 2.0  # Profile constant
        
        downstream_face = stations > 0
        
        # Apply ogee equation to downstream face
        # z = crest_elevation - (x^1.85 / (K * H^0.85))
        x_ogee = stations[downstream_face]
        y_ogee = (x_ogee**1.85) / (k * self.design_head**0.85)
        
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
        """
        if head <= 0:
            return 0
        
        # Base coefficient at design head
        design_coefficient = 0.75
        
        # Head ratio (actual/design)
        head_ratio = head / self.design_head
        
        if head_ratio <= 0.3:
            # For low head ratios
            return 0.65
        elif head_ratio <= 0.6:
            # Linear interpolation for intermediate values
            return 0.65 + (0.75 - 0.65) * (head_ratio - 0.3) / 0.3
        elif head_ratio <= 1.0:
            # Near design head
            return 0.75
        elif head_ratio <= 1.5:
            # Above design head
            return 0.75 + (0.85 - 0.75) * (head_ratio - 1.0) / 0.5
        else:
            # Well above design head
            return 0.85
    
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
        
        # Q = Cd * L * H^(3/2) * sqrt(2g)
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
        
        # Initial guess for head
        head_guess = (discharge / (0.75 * width * math.sqrt(2 * 9.81)))**(2/3)
        
        # Iterative solution (similar to other weir types)
        for i in range(max_iterations):
            calculated_discharge = self.calculate_discharge(head_guess, width)
            
            if abs(calculated_discharge - discharge) < tolerance:
                break
            
            # Adjust head guess
            h_pow = head_guess**(0.5)
            if h_pow < 1e-10:
                h_pow = 1e-10
                
            dQ_dH = 1.5 * self.get_discharge_coefficient(head_guess) * width * h_pow * math.sqrt(2 * 9.81)
            damping = 0.7
            correction = (calculated_discharge - discharge) / dQ_dH
            head_guess -= damping * correction
            
            if head_guess <= 0:
                head_guess = 0.01
        
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
        
        return OgeeWeir(kwargs['height'], kwargs['crest_elevation'], kwargs['design_head'])
    
    else:
        raise ValueError(f"Unknown dam type: {dam_type}")