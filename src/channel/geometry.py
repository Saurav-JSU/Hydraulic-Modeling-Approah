"""
Channel geometry definitions and calculations.

This module provides classes and functions for defining different channel
cross-sections and calculating their geometric properties.
"""

import math
from typing import Dict, Optional, Union, List, Callable
import numpy as np

class Channel:
    """Base class for all channel types."""
    
    def __init__(self, roughness: float):
        """
        Initialize a channel with roughness coefficient.
        
        Parameters:
            roughness (float): Manning's roughness coefficient
        """
        if roughness <= 0:
            raise ValueError("Roughness coefficient must be positive")
        
        self.roughness = roughness
    
    def area(self, depth: float) -> float:
        """
        Calculate cross-sectional flow area.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Flow area (m²)
        """
        raise NotImplementedError("Subclasses must implement area()")
    
    def wetted_perimeter(self, depth: float) -> float:
        """
        Calculate wetted perimeter.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Wetted perimeter (m)
        """
        raise NotImplementedError("Subclasses must implement wetted_perimeter()")
    
    def top_width(self, depth: float) -> float:
        """
        Calculate top width of water surface.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Top width (m)
        """
        raise NotImplementedError("Subclasses must implement top_width()")
    
    def hydraulic_radius(self, depth: float) -> float:
        """
        Calculate hydraulic radius.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Hydraulic radius (m)
        """
        if depth <= 0:
            return 0
        return self.area(depth) / self.wetted_perimeter(depth)
    
    def hydraulic_depth(self, depth: float) -> float:
        """
        Calculate hydraulic depth.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Hydraulic depth (m)
        """
        if depth <= 0:
            return 0
        return self.area(depth) / self.top_width(depth)
    
    def section_factor(self, depth: float) -> float:
        """
        Calculate section factor (AR^(2/3)) for Manning's equation.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Section factor (m^(8/3))
        """
        if depth <= 0:
            return 0
        return self.area(depth) * (self.hydraulic_radius(depth) ** (2/3))
    
    def conveyance(self, depth: float) -> float:
        """
        Calculate channel conveyance (K = (1/n)AR^(2/3)).
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Conveyance (m³/s when multiplied by sqrt(S))
        """
        return (1.0 / self.roughness) * self.section_factor(depth)
    
    def critical_depth(self, discharge: float, g: float = 9.81, 
                       depth_range: tuple = (0.01, 10), 
                       tolerance: float = 1e-6, 
                       max_iterations: int = 100) -> float:
        """
        Calculate critical depth using numerical method.
        
        Parameters:
            discharge (float): Flow rate (m³/s)
            g (float): Gravitational acceleration (m/s²)
            depth_range (tuple): Range of depths to search (m)
            tolerance (float): Convergence tolerance
            max_iterations (int): Maximum number of iterations
            
        Returns:
            float: Critical depth (m)
            
        Raises:
            ValueError: If discharge is negative
            RuntimeError: If solution doesn't converge
        """
        if discharge < 0:
            raise ValueError("Discharge must be non-negative")
        
        if discharge == 0:
            return 0
        
        # Critical flow condition: Q²T/gA³ = 1
        def critical_flow_function(y: float) -> float:
            if y <= 0:
                return float('inf')
            
            A = self.area(y)
            T = self.top_width(y)
            
            if A <= 0 or T <= 0:
                return float('inf')
            
            return (discharge**2 * T) / (g * A**3) - 1
        
        # Use bisection method to find critical depth
        y_min, y_max = depth_range
        
        # Check if bounds are valid
        f_min = critical_flow_function(y_min)
        f_max = critical_flow_function(y_max)
        
        if f_min * f_max > 0:
            # If same sign, adjust the range
            if f_min < 0:
                y_max = y_max * 10
            else:
                y_min = y_min / 10
        
        # Bisection method
        for i in range(max_iterations):
            y_mid = (y_min + y_max) / 2
            f_mid = critical_flow_function(y_mid)
            
            if abs(f_mid) < tolerance:
                return y_mid
            
            if f_mid * f_min < 0:
                y_max = y_mid
            else:
                y_min = y_mid
                f_min = f_mid
            
            if y_max - y_min < tolerance:
                return y_mid
        
        raise RuntimeError(f"Critical depth calculation did not converge within {max_iterations} iterations")


class RectangularChannel(Channel):
    """Rectangular channel cross-section."""
    
    def __init__(self, bottom_width: float, roughness: float):
        """
        Initialize a rectangular channel.
        
        Parameters:
            bottom_width (float): Channel bottom width (m)
            roughness (float): Manning's roughness coefficient
        """
        super().__init__(roughness)
        
        if bottom_width <= 0:
            raise ValueError("Bottom width must be positive")
        
        self.bottom_width = bottom_width
    
    def area(self, depth: float) -> float:
        """
        Calculate cross-sectional flow area.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Flow area (m²)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        return self.bottom_width * depth
    
    def wetted_perimeter(self, depth: float) -> float:
        """
        Calculate wetted perimeter.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Wetted perimeter (m)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        if depth == 0:
            return self.bottom_width
        
        return self.bottom_width + 2 * depth
    
    def top_width(self, depth: float) -> float:
        """
        Calculate top width of water surface.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Top width (m)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        return self.bottom_width
    
    def critical_depth(self, discharge: float, g: float = 9.81) -> float:
        """
        Calculate critical depth for rectangular channel.
        
        Parameters:
            discharge (float): Flow rate (m³/s)
            g (float): Gravitational acceleration (m/s²)
            
        Returns:
            float: Critical depth (m)
            
        Raises:
            ValueError: If discharge is negative
        """
        if discharge < 0:
            raise ValueError("Discharge must be non-negative")
        
        if discharge == 0:
            return 0
        
        # Critical depth for rectangular channel: yc = (q²/g)^(1/3)
        unit_discharge = discharge / self.bottom_width
        return (unit_discharge**2 / g)**(1/3)


class TrapezoidalChannel(Channel):
    """Trapezoidal channel cross-section."""
    
    def __init__(self, bottom_width: float, side_slope: float, roughness: float):
        """
        Initialize a trapezoidal channel.
        
        Parameters:
            bottom_width (float): Channel bottom width (m)
            side_slope (float): Side slope (horizontal/vertical)
            roughness (float): Manning's roughness coefficient
        """
        super().__init__(roughness)
        
        if bottom_width < 0:
            raise ValueError("Bottom width must be non-negative")
        if side_slope < 0:
            raise ValueError("Side slope must be non-negative")
        
        self.bottom_width = bottom_width
        self.side_slope = side_slope
    
    def area(self, depth: float) -> float:
        """
        Calculate cross-sectional flow area.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Flow area (m²)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        # Area = (bottom width + top width) * depth / 2
        top_width = self.bottom_width + 2 * self.side_slope * depth
        return (self.bottom_width + top_width) * depth / 2
    
    def wetted_perimeter(self, depth: float) -> float:
        """
        Calculate wetted perimeter.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Wetted perimeter (m)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        if depth == 0:
            return self.bottom_width
        
        # Wetted perimeter = bottom width + 2 * sloped sides
        sloped_side_length = depth * math.sqrt(1 + self.side_slope**2)
        return self.bottom_width + 2 * sloped_side_length
    
    def top_width(self, depth: float) -> float:
        """
        Calculate top width of water surface.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Top width (m)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        return self.bottom_width + 2 * self.side_slope * depth


class TriangularChannel(TrapezoidalChannel):
    """Triangular channel cross-section."""
    
    def __init__(self, side_slope: float, roughness: float):
        """
        Initialize a triangular channel.
        
        Parameters:
            side_slope (float): Side slope (horizontal/vertical)
            roughness (float): Manning's roughness coefficient
        """
        # Triangular channel is a special case of trapezoidal with zero bottom width
        super().__init__(0, side_slope, roughness)


class CompoundChannel(Channel):
    """Compound channel with main channel and floodplains."""
    
    def __init__(self, main_channel: Channel, floodplains: List[Channel], 
                main_channel_depth: float):
        """
        Initialize a compound channel.
        
        Parameters:
            main_channel (Channel): Main channel
            floodplains (List[Channel]): List of floodplain channels
            main_channel_depth (float): Bankfull depth of main channel
        """
        # Use main channel's roughness as default
        super().__init__(main_channel.roughness)
        
        if main_channel_depth <= 0:
            raise ValueError("Main channel depth must be positive")
        
        self.main_channel = main_channel
        self.floodplains = floodplains
        self.main_channel_depth = main_channel_depth
    
    def area(self, depth: float) -> float:
        """
        Calculate cross-sectional flow area.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Flow area (m²)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        if depth <= self.main_channel_depth:
            # Only main channel flow
            return self.main_channel.area(depth)
        else:
            # Main channel + floodplains
            main_area = self.main_channel.area(self.main_channel_depth)
            
            # Calculate floodplain contribution
            floodplain_depth = depth - self.main_channel_depth
            floodplain_area = sum(fp.area(floodplain_depth) for fp in self.floodplains)
            
            return main_area + floodplain_area
    
    def wetted_perimeter(self, depth: float) -> float:
        """
        Calculate wetted perimeter.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Wetted perimeter (m)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        if depth <= self.main_channel_depth:
            # Only main channel flow
            return self.main_channel.wetted_perimeter(depth)
        else:
            # Main channel + floodplains
            main_wp = self.main_channel.wetted_perimeter(self.main_channel_depth)
            
            # Subtract top width from main channel (interface with floodplains)
            main_wp -= self.main_channel.top_width(self.main_channel_depth)
            
            # Calculate floodplain contribution
            floodplain_depth = depth - self.main_channel_depth
            floodplain_wp = sum(fp.wetted_perimeter(floodplain_depth) for fp in self.floodplains)
            
            # Subtract bottom widths of floodplains (interface with main channel)
            for fp in self.floodplains:
                if isinstance(fp, RectangularChannel) or isinstance(fp, TrapezoidalChannel):
                    floodplain_wp -= fp.bottom_width
            
            return main_wp + floodplain_wp
    
    def top_width(self, depth: float) -> float:
        """
        Calculate top width of water surface.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Top width (m)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        if depth <= self.main_channel_depth:
            # Only main channel flow
            return self.main_channel.top_width(depth)
        else:
            # Main channel + floodplains
            floodplain_depth = depth - self.main_channel_depth
            return sum(fp.top_width(floodplain_depth) for fp in self.floodplains)
    
    def conveyance(self, depth: float) -> float:
        """
        Calculate channel conveyance using divided channel method.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Conveyance (m³/s when multiplied by sqrt(S))
        """
        if depth <= 0:
            return 0
        
        if depth <= self.main_channel_depth:
            # Only main channel flow
            return self.main_channel.conveyance(depth)
        else:
            # Main channel at bankfull + floodplains
            main_k = self.main_channel.conveyance(self.main_channel_depth)
            
            # Calculate floodplain contribution
            floodplain_depth = depth - self.main_channel_depth
            floodplain_k = sum(fp.conveyance(floodplain_depth) for fp in self.floodplains)
            
            return main_k + floodplain_k


class CircularChannel(Channel):
    """Circular channel (pipe) cross-section."""
    
    def __init__(self, diameter: float, roughness: float):
        """
        Initialize a circular channel.
        
        Parameters:
            diameter (float): Pipe diameter (m)
            roughness (float): Manning's roughness coefficient
        """
        super().__init__(roughness)
        
        if diameter <= 0:
            raise ValueError("Diameter must be positive")
        
        self.diameter = diameter
        self.radius = diameter / 2
    
    def _get_angle(self, depth: float) -> float:
        """Calculate central angle for partially filled circular section."""
        if depth <= 0:
            return 0
        if depth >= self.diameter:
            return 2 * math.pi
        
        # Central angle calculation
        h = depth
        r = self.radius
        
        if h <= r:
            # Less than half full
            return 2 * math.acos((r - h) / r)
        else:
            # More than half full
            return 2 * math.pi - 2 * math.acos((h - r) / r)
    
    def area(self, depth: float) -> float:
        """
        Calculate cross-sectional flow area.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Flow area (m²)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        if depth >= self.diameter:
            # Full pipe
            return math.pi * self.radius**2
        
        if depth == 0:
            return 0
        
        # Partially filled pipe
        theta = self._get_angle(depth)
        return (self.radius**2 / 2) * (theta - math.sin(theta))
    
    def wetted_perimeter(self, depth: float) -> float:
        """
        Calculate wetted perimeter.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Wetted perimeter (m)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        if depth >= self.diameter:
            # Full pipe
            return math.pi * self.diameter
        
        if depth == 0:
            return 0
        
        # Partially filled pipe
        theta = self._get_angle(depth)
        return self.radius * theta
    
    def top_width(self, depth: float) -> float:
        """
        Calculate top width of water surface.
        
        Parameters:
            depth (float): Water depth (m)
            
        Returns:
            float: Top width (m)
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        if depth >= self.diameter:
            # Full pipe - no free surface
            return 0
        
        if depth == 0:
            return 0
        
        # Partially filled pipe
        return 2 * self.radius * math.sin(self._get_angle(depth) / 2)


def create_channel(channel_type: str, **kwargs) -> Channel:
    """
    Factory function to create channel objects.
    
    Parameters:
        channel_type (str): Type of channel ('rectangular', 'trapezoidal', 'triangular', 'circular')
        **kwargs: Channel parameters
        
    Returns:
        Channel: Channel object
        
    Raises:
        ValueError: If channel type is unknown
    """
    channel_type = channel_type.lower()
    
    if channel_type == 'rectangular':
        required = {'bottom_width', 'roughness'}
        if not required.issubset(kwargs.keys()):
            missing = required - set(kwargs.keys())
            raise ValueError(f"Missing parameters for rectangular channel: {missing}")
        
        return RectangularChannel(kwargs['bottom_width'], kwargs['roughness'])
    
    elif channel_type == 'trapezoidal':
        required = {'bottom_width', 'side_slope', 'roughness'}
        if not required.issubset(kwargs.keys()):
            missing = required - set(kwargs.keys())
            raise ValueError(f"Missing parameters for trapezoidal channel: {missing}")
        
        return TrapezoidalChannel(kwargs['bottom_width'], kwargs['side_slope'], kwargs['roughness'])
    
    elif channel_type == 'triangular':
        required = {'side_slope', 'roughness'}
        if not required.issubset(kwargs.keys()):
            missing = required - set(kwargs.keys())
            raise ValueError(f"Missing parameters for triangular channel: {missing}")
        
        return TriangularChannel(kwargs['side_slope'], kwargs['roughness'])
    
    elif channel_type == 'circular':
        required = {'diameter', 'roughness'}
        if not required.issubset(kwargs.keys()):
            missing = required - set(kwargs.keys())
            raise ValueError(f"Missing parameters for circular channel: {missing}")
        
        return CircularChannel(kwargs['diameter'], kwargs['roughness'])
    
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")