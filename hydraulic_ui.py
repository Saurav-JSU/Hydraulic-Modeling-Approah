#!/usr/bin/env python3
"""
Interactive Hydraulic Modeling Interface

A PyQt-based application for hydraulic modeling of dams and channels,
with integrated visualization capabilities.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QTabWidget, 
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, 
    QScrollArea, QSplitter, QFrame, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont, QDoubleValidator, QPixmap

# Add the src directory to the Python path - adjust as needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the necessary modules from your hydraulic modeling code
try:
    from src.dam.geometry import BroadCrestedWeir, SharpCrestedWeir, OgeeWeir, create_dam
    from src.channel.geometry import RectangularChannel, TrapezoidalChannel, create_channel
    from src.dam.flow import calculate_flow_over_dam, create_rating_curve, calculate_energy_dissipation
    from src.dam.flow import estimate_tailwater_profile, hydraulic_jump_location
    from src.channel.normal_flow import normal_depth, critical_depth
    from src.hydraulics.energy import specific_energy
    
    # Import visualization modules
    from src.visualization import (
        plot_enhanced_profile, 
        create_flow_regime_dashboard,
        plot_channel_cross_section,
        create_profile_with_parameter_plots,
        visualize_hydraulic_jump
    )
    
    # Import successful - use the existing modules
    USING_EXISTING_MODULES = True
    
except ImportError as e:
    print(f"Warning: Could not import hydraulic modules: {e}")
    print("Creating standalone mode with simplified calculations")
    USING_EXISTING_MODULES = False
    
    # We'll define simplified placeholders for testing the UI
    # These will be replaced with proper implementation if the modules are available


class HydraulicModelingApp(QMainWindow):
    """Main application window for the hydraulic modeling interface."""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Hydraulic Modeling Interface")
        self.setMinimumSize(1200, 800)
        
        # Current calculation results
        self.results = None
        self.scenario = None
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Create the central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create a splitter to separate parameters and visualization
        splitter = QSplitter(Qt.Horizontal)
        
        # ------ Parameter Panel ------
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)
        param_panel.setMaximumWidth(400)
        
        # --- Dam Parameters ---
        dam_group = QGroupBox("Dam Parameters")
        dam_layout = QFormLayout()
        
        # Dam type selection
        self.dam_type_combo = QComboBox()
        self.dam_type_combo.addItems(["Ogee Spillway", "Broad-Crested Weir", "Sharp-Crested Weir"])
        self.dam_type_combo.currentIndexChanged.connect(self.on_dam_type_changed)
        dam_layout.addRow("Dam Type:", self.dam_type_combo)
        
        # Dam height
        self.dam_height_spin = QDoubleSpinBox()
        self.dam_height_spin.setRange(1, 50)
        self.dam_height_spin.setValue(10.0)
        self.dam_height_spin.setSuffix(" m")
        dam_layout.addRow("Dam Height:", self.dam_height_spin)
        
        # Dam crest elevation
        self.dam_crest_spin = QDoubleSpinBox()
        self.dam_crest_spin.setRange(0, 500)
        self.dam_crest_spin.setValue(100.0)
        self.dam_crest_spin.setSuffix(" m")
        dam_layout.addRow("Crest Elevation:", self.dam_crest_spin)
        
        # Ogee design head
        self.ogee_design_head_spin = QDoubleSpinBox()
        self.ogee_design_head_spin.setRange(0.1, 10)
        self.ogee_design_head_spin.setValue(2.0)
        self.ogee_design_head_spin.setSuffix(" m")
        dam_layout.addRow("Design Head:", self.ogee_design_head_spin)
        
        # Broad-crested weir crest width
        self.bc_crest_width_spin = QDoubleSpinBox()
        self.bc_crest_width_spin.setRange(0.1, 10)
        self.bc_crest_width_spin.setValue(3.0)
        self.bc_crest_width_spin.setSuffix(" m")
        dam_layout.addRow("Crest Width:", self.bc_crest_width_spin)
        self.bc_crest_width_spin.setVisible(False)
        
        dam_group.setLayout(dam_layout)
        param_layout.addWidget(dam_group)
        
        # --- Channel Parameters ---
        channel_group = QGroupBox("Channel Parameters")
        channel_layout = QFormLayout()
        
        # Channel type selection
        self.channel_type_combo = QComboBox()
        self.channel_type_combo.addItems(["Rectangular", "Trapezoidal"])
        self.channel_type_combo.currentIndexChanged.connect(self.on_channel_type_changed)
        channel_layout.addRow("Channel Type:", self.channel_type_combo)
        
        # Channel bottom width
        self.channel_width_spin = QDoubleSpinBox()
        self.channel_width_spin.setRange(0.5, 50)
        self.channel_width_spin.setValue(5.0)
        self.channel_width_spin.setSuffix(" m")
        channel_layout.addRow("Bottom Width:", self.channel_width_spin)
        
        # Channel side slope (for trapezoidal)
        self.side_slope_spin = QDoubleSpinBox()
        self.side_slope_spin.setRange(0, 10)
        self.side_slope_spin.setValue(1.5)
        self.side_slope_spin.setSuffix(" H:1V")
        channel_layout.addRow("Side Slope:", self.side_slope_spin)
        
        # Channel roughness (Manning's n)
        self.roughness_spin = QDoubleSpinBox()
        self.roughness_spin.setRange(0.01, 0.1)
        self.roughness_spin.setValue(0.015)
        self.roughness_spin.setDecimals(3)
        self.roughness_spin.setSingleStep(0.005)
        channel_layout.addRow("Manning's n:", self.roughness_spin)
        
        # Channel slope
        self.channel_slope_spin = QDoubleSpinBox()
        self.channel_slope_spin.setRange(0.0001, 0.1)
        self.channel_slope_spin.setValue(0.001)
        self.channel_slope_spin.setDecimals(4)
        self.channel_slope_spin.setSingleStep(0.001)
        channel_layout.addRow("Channel Slope:", self.channel_slope_spin)
        
        # Downstream slope
        self.downstream_slope_spin = QDoubleSpinBox()
        self.downstream_slope_spin.setRange(0.0001, 0.1)
        self.downstream_slope_spin.setValue(0.005)
        self.downstream_slope_spin.setDecimals(4)
        self.downstream_slope_spin.setSingleStep(0.001)
        channel_layout.addRow("Downstream Slope:", self.downstream_slope_spin)
        
        # Channel width at dam
        self.width_at_dam_spin = QDoubleSpinBox()
        self.width_at_dam_spin.setRange(1, 100)
        self.width_at_dam_spin.setValue(20.0)
        self.width_at_dam_spin.setSuffix(" m")
        channel_layout.addRow("Width at Dam:", self.width_at_dam_spin)
        
        channel_group.setLayout(channel_layout)
        param_layout.addWidget(channel_group)
        
        # --- Flow Parameters ---
        flow_group = QGroupBox("Flow Parameters")
        flow_layout = QFormLayout()
        
        # Initial upstream water level
        self.initial_level_spin = QDoubleSpinBox()
        self.initial_level_spin.setRange(0, 200)
        self.initial_level_spin.setValue(95.0)
        self.initial_level_spin.setSuffix(" m")
        flow_layout.addRow("Initial Water Level:", self.initial_level_spin)
        
        # Flood peak water level
        self.flood_level_spin = QDoubleSpinBox()
        self.flood_level_spin.setRange(0, 200)
        self.flood_level_spin.setValue(102.0)
        self.flood_level_spin.setSuffix(" m")
        flow_layout.addRow("Flood Peak Level:", self.flood_level_spin)
        
        # Current water level (for analysis)
        self.current_level_spin = QDoubleSpinBox()
        self.current_level_spin.setRange(0, 200)
        self.current_level_spin.setValue(101.0)
        self.current_level_spin.setSuffix(" m")
        flow_layout.addRow("Analysis Water Level:", self.current_level_spin)
        
        flow_group.setLayout(flow_layout)
        param_layout.addWidget(flow_group)
        
        # --- Action Buttons ---
        button_layout = QHBoxLayout()
        
        # Calculate button
        self.calc_button = QPushButton("Calculate")
        self.calc_button.clicked.connect(self.on_calculate)
        button_layout.addWidget(self.calc_button)
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset)
        button_layout.addWidget(self.reset_button)
        
        # Export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.on_export)
        self.export_button.setEnabled(False)  # Disabled until results exist
        button_layout.addWidget(self.export_button)
        
        param_layout.addLayout(button_layout)
        
        # Add stretch to push everything to the top
        param_layout.addStretch()
        
        # ------ Visualization Panel ------
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        
        # Create tabs for different visualization types
        viz_tabs = QTabWidget()
        
        # Profile tab
        self.profile_tab = QWidget()
        profile_layout = QVBoxLayout(self.profile_tab)
        
        # Create matplotlib figure for the profile
        self.profile_figure = Figure(figsize=(8, 6), dpi=100)
        self.profile_canvas = FigureCanvas(self.profile_figure)
        self.profile_toolbar = NavigationToolbar(self.profile_canvas, self)
        
        profile_layout.addWidget(self.profile_toolbar)
        profile_layout.addWidget(self.profile_canvas)
        
        viz_tabs.addTab(self.profile_tab, "Profile View")
        
        # Cross-section tab
        self.cross_tab = QWidget()
        cross_layout = QVBoxLayout(self.cross_tab)
        
        # Controls for cross-section location
        cross_control_layout = QHBoxLayout()
        cross_control_layout.addWidget(QLabel("Location:"))
        
        self.cross_location_spin = QDoubleSpinBox()
        self.cross_location_spin.setRange(-50, 500)
        self.cross_location_spin.setValue(0)
        self.cross_location_spin.setSuffix(" m")
        self.cross_location_spin.valueChanged.connect(self.update_cross_section)
        cross_control_layout.addWidget(self.cross_location_spin)
        
        # Parameter to highlight
        cross_control_layout.addWidget(QLabel("Highlight:"))
        self.cross_param_combo = QComboBox()
        self.cross_param_combo.addItems(["Froude Number", "Velocity", "Shear Stress", "None"])
        self.cross_param_combo.currentIndexChanged.connect(self.update_cross_section)
        cross_control_layout.addWidget(self.cross_param_combo)
        
        cross_layout.addLayout(cross_control_layout)
        
        # Create matplotlib figure for the cross-section
        self.cross_figure = Figure(figsize=(8, 6), dpi=100)
        self.cross_canvas = FigureCanvas(self.cross_figure)
        self.cross_toolbar = NavigationToolbar(self.cross_canvas, self)
        
        cross_layout.addWidget(self.cross_toolbar)
        cross_layout.addWidget(self.cross_canvas)
        
        viz_tabs.addTab(self.cross_tab, "Cross-Section View")
        
        # Flow regime tab
        self.regime_tab = QWidget()
        regime_layout = QVBoxLayout(self.regime_tab)
        
        # Create matplotlib figure for flow regimes
        self.regime_figure = Figure(figsize=(8, 6), dpi=100)
        self.regime_canvas = FigureCanvas(self.regime_figure)
        self.regime_toolbar = NavigationToolbar(self.regime_canvas, self)
        
        regime_layout.addWidget(self.regime_toolbar)
        regime_layout.addWidget(self.regime_canvas)
        
        viz_tabs.addTab(self.regime_tab, "Flow Regimes")
        
        # Results tab for text output
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        
        self.results_text = QLabel("No results calculated yet.")
        self.results_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.results_text.setWordWrap(True)
        self.results_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        results_scroll = QScrollArea()
        results_scroll.setWidget(self.results_text)
        results_scroll.setWidgetResizable(True)
        
        results_layout.addWidget(results_scroll)
        
        viz_tabs.addTab(self.results_tab, "Results Summary")
        
        viz_layout.addWidget(viz_tabs)
        
        # Add panels to the splitter
        splitter.addWidget(param_panel)
        splitter.addWidget(viz_panel)
        
        # Set the splitter as the central widget
        layout = QHBoxLayout(central_widget)
        layout.addWidget(splitter)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Initialize UI state
        self.on_dam_type_changed()
        self.on_channel_type_changed()
        
        # Set up initial empty plots
        self.setup_empty_plots()
        
        # Show the window
        self.show()
    
    def setup_empty_plots(self):
        """Set up initial empty plots."""
        # Profile plot
        self.profile_figure.clear()
        ax = self.profile_figure.add_subplot(111)
        ax.set_xlabel('Distance from Dam (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Hydraulic Profile View')
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "Click 'Calculate' to generate results", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, style='italic', color='gray')
        self.profile_canvas.draw()
        
        # Cross-section plot
        self.cross_figure.clear()
        ax = self.cross_figure.add_subplot(111)
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Channel Cross-Section View')
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "Click 'Calculate' to generate results", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, style='italic', color='gray')
        ax.set_aspect('equal')
        self.cross_canvas.draw()
        
        # Flow regime plot
        self.regime_figure.clear()
        ax = self.regime_figure.add_subplot(111)
        ax.set_xlabel('Distance from Dam (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Flow Regime Visualization')
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "Click 'Calculate' to generate results", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, style='italic', color='gray')
        self.regime_canvas.draw()
    
    def on_dam_type_changed(self):
        """Handle dam type selection change."""
        dam_type = self.dam_type_combo.currentText()
        
        # Show/hide parameters based on dam type
        if dam_type == "Broad-Crested Weir":
            self.ogee_design_head_spin.setVisible(False)
            self.bc_crest_width_spin.setVisible(True)
        elif dam_type == "Ogee Spillway":
            self.ogee_design_head_spin.setVisible(True)
            self.bc_crest_width_spin.setVisible(False)
        else:  # Sharp-Crested Weir
            self.ogee_design_head_spin.setVisible(False)
            self.bc_crest_width_spin.setVisible(False)
    
    def on_channel_type_changed(self):
        """Handle channel type selection change."""
        channel_type = self.channel_type_combo.currentText()
        
        # Show/hide parameters based on channel type
        if channel_type == "Trapezoidal":
            self.side_slope_spin.setVisible(True)
        else:  # Rectangular
            self.side_slope_spin.setVisible(False)
    
    def on_calculate(self):
        """Handle calculate button click."""
        try:
            # Create a scenario dictionary from the UI parameters
            self.scenario = self.create_scenario_from_ui()
            
            # Get the water level for analysis
            water_level = self.current_level_spin.value()
            
            # Calculate hydraulic results
            self.results = self.analyze_steady_state(self.scenario, water_level)
            
            # Update visualizations
            self.update_profile_view()
            self.update_regime_view()
            self.update_cross_section()
            self.update_results_text()
            
            # Enable export button
            self.export_button.setEnabled(True)
            
            # Show success message
            QMessageBox.information(self, "Calculation Complete", 
                                  "Hydraulic calculations completed successfully.")
            
        except Exception as e:
            # Show error message
            QMessageBox.critical(self, "Calculation Error", 
                                f"An error occurred during calculation:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_reset(self):
        """Handle reset button click."""
        # Reset results
        self.results = None
        self.scenario = None
        
        # Reset visualizations
        self.setup_empty_plots()
        self.results_text.setText("No results calculated yet.")
        
        # Disable export button
        self.export_button.setEnabled(False)
    
    def on_export(self):
        """Handle export button click."""
        if not self.results:
            return
        
        # Create a file dialog to get save location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)",
            options=options
        )
        
        if file_name:
            try:
                # Determine file type from extension
                ext = os.path.splitext(file_name)[1].lower()
                
                if ext == '.png':
                    # Save the current tab's figure as PNG
                    current_tab = self.centralWidget().findChild(QTabWidget).currentWidget()
                    
                    if current_tab == self.profile_tab:
                        self.profile_figure.savefig(file_name, dpi=300, bbox_inches='tight')
                    elif current_tab == self.cross_tab:
                        self.cross_figure.savefig(file_name, dpi=300, bbox_inches='tight')
                    elif current_tab == self.regime_tab:
                        self.regime_figure.savefig(file_name, dpi=300, bbox_inches='tight')
                    else:
                        # Results tab - create a text file instead
                        with open(file_name, 'w') as f:
                            f.write(self.results_text.text())
                
                elif ext == '.pdf':
                    # Save the current tab's figure as PDF
                    current_tab = self.centralWidget().findChild(QTabWidget).currentWidget()
                    
                    if current_tab == self.profile_tab:
                        self.profile_figure.savefig(file_name, format='pdf', bbox_inches='tight')
                    elif current_tab == self.cross_tab:
                        self.cross_figure.savefig(file_name, format='pdf', bbox_inches='tight')
                    elif current_tab == self.regime_tab:
                        self.regime_figure.savefig(file_name, format='pdf', bbox_inches='tight')
                    else:
                        # Results tab - create a text file instead
                        text_file = os.path.splitext(file_name)[0] + '.txt'
                        with open(text_file, 'w') as f:
                            f.write(self.results_text.text())
                
                # Show success message
                QMessageBox.information(self, "Export Complete", 
                                      f"Results exported to {file_name}")
                
            except Exception as e:
                # Show error message
                QMessageBox.critical(self, "Export Error", 
                                   f"An error occurred during export:\n{str(e)}")
    
    def create_scenario_from_ui(self):
        """Create a scenario dictionary from UI parameters."""
        # Get dam parameters
        dam_type = self.dam_type_combo.currentText()
        dam_height = self.dam_height_spin.value()
        dam_crest = self.dam_crest_spin.value()
        
        # Create the appropriate dam object based on type
        if USING_EXISTING_MODULES:
            if dam_type == "Ogee Spillway":
                design_head = self.ogee_design_head_spin.value()
                dam = OgeeWeir(height=dam_height, crest_elevation=dam_crest, 
                              design_head=design_head)
                dam_params = {'design_head': design_head}
            elif dam_type == "Broad-Crested Weir":
                crest_width = self.bc_crest_width_spin.value()
                dam = BroadCrestedWeir(height=dam_height, crest_elevation=dam_crest,
                                      crest_width=crest_width)
                dam_params = {'crest_width': crest_width}
            else:  # Sharp-Crested Weir
                dam = SharpCrestedWeir(height=dam_height, crest_elevation=dam_crest)
                dam_params = {}
        else:
            # Placeholder if modules not available
            dam = None
            dam_params = {
                'type': dam_type.lower().replace('-', '_').replace(' ', '_'),
                'height': dam_height,
                'crest_elevation': dam_crest
            }
            if dam_type == "Ogee Spillway":
                dam_params['design_head'] = self.ogee_design_head_spin.value()
            elif dam_type == "Broad-Crested Weir":
                dam_params['crest_width'] = self.bc_crest_width_spin.value()
        
        # Get channel parameters
        channel_type = self.channel_type_combo.currentText()
        bottom_width = self.channel_width_spin.value()
        roughness = self.roughness_spin.value()
        channel_slope = self.channel_slope_spin.value()
        downstream_slope = self.downstream_slope_spin.value()
        
        # Create the appropriate channel objects based on type
        if USING_EXISTING_MODULES:
            if channel_type == "Rectangular":
                upstream_channel = RectangularChannel(bottom_width=bottom_width, 
                                                    roughness=roughness)
                downstream_channel = RectangularChannel(bottom_width=bottom_width, 
                                                      roughness=roughness)
                side_slope = 0
            else:  # Trapezoidal
                side_slope = self.side_slope_spin.value()
                upstream_channel = TrapezoidalChannel(bottom_width=bottom_width,
                                                    side_slope=side_slope,
                                                    roughness=roughness)
                downstream_channel = TrapezoidalChannel(bottom_width=bottom_width,
                                                      side_slope=side_slope,
                                                      roughness=roughness)
        else:
            # Placeholder if modules not available
            upstream_channel = None
            downstream_channel = None
            if channel_type == "Rectangular":
                side_slope = 0
            else:  # Trapezoidal
                side_slope = self.side_slope_spin.value()
        
        # Get flow parameters
        initial_water_level = self.initial_level_spin.value()
        flood_water_level = self.flood_level_spin.value()
        
        # Create the scenario dictionary
        scenario = {
            # Objects
            'dam': dam,
            'upstream_channel': upstream_channel,
            'downstream_channel': downstream_channel,
            
            # Dam parameters
            'dam_type': dam_type.lower().replace('-', '_').replace(' ', '_'),
            'dam_height': dam_height,
            'dam_crest_elevation': dam_crest,
            'dam_base_elevation': dam_crest - dam_height,
            **dam_params,
            
            # Channel parameters
            'channel_type': channel_type.lower(),
            'channel_bottom_width': bottom_width,
            'channel_side_slope': side_slope,
            'channel_roughness': roughness,
            'channel_slope': channel_slope,
            'downstream_slope': downstream_slope,
            'channel_width_at_dam': self.width_at_dam_spin.value(),
            
            # Water levels
            'initial_water_level': initial_water_level,
            'flood_water_level': flood_water_level
        }
        
        return scenario
    
    def analyze_steady_state(self, scenario, upstream_level=None):
        """Analyze the steady-state hydraulic conditions."""
        if not USING_EXISTING_MODULES:
            # Create simplified placeholder results for testing the UI
            return self.create_placeholder_results(scenario, upstream_level)
        
        # Get scenario parameters
        dam = scenario['dam']
        downstream_channel = scenario['downstream_channel']
        channel_width = scenario['channel_width_at_dam']
        downstream_slope = scenario['downstream_slope']
        
        # Use provided upstream level or default to flood peak
        if upstream_level is None:
            upstream_level = scenario['flood_water_level']
        
        # Check if water level is above crest
        if upstream_level <= scenario['dam_crest_elevation']:
            # No flow over dam - return minimal results
            return {
                'upstream_level': upstream_level,
                'discharge': 0,
                'head': 0,
                'velocity': 0,
                'energy_results': {'energy_loss': 0, 'dissipation_ratio': 0, 'power': 0},
                'tailwater': {'x': np.array([0, 100]), 
                            'z': np.array([scenario['dam_base_elevation'], 
                                           scenario['dam_base_elevation'] - 5]), 
                            'wse': np.array([scenario['dam_base_elevation'], 
                                            scenario['dam_base_elevation'] - 5]), 
                            'y': np.array([0, 0]), 
                            'v': np.array([0, 0]), 
                            'fr': np.array([0, 0])},
                'normal_depth': 0,
                'critical_depth': 0,
                'hydraulic_jump': {'jump_possible': False, 'reason': 'No flow over dam'}
            }
        
        # Calculate flow over dam
        # Set minimum downstream water level to avoid division by zero
        minimum_depth = 0.1  # m, minimum depth to avoid division by zero
        downstream_level = scenario['dam_base_elevation'] + minimum_depth
        
        flow_results = calculate_flow_over_dam(
            dam, upstream_level, downstream_level, channel_width
        )
        
        discharge = flow_results['discharge']
        head = flow_results['head']
        velocity = flow_results['velocity']
        
        # Calculate downstream normal depth
        yn = normal_depth(downstream_channel, discharge, downstream_slope)
        yc = critical_depth(downstream_channel, discharge)
        
        # Calculate tailwater profile
        tailwater = estimate_tailwater_profile(
            dam, discharge, channel_width, downstream_slope,
            downstream_channel.roughness, 500, 100
        )
        
        # Update downstream level to use the first calculated depth from tailwater
        # This provides a more realistic value for energy calculations
        if len(tailwater['y']) > 0:
            realistic_downstream_depth = tailwater['y'][0]
            downstream_level = scenario['dam_base_elevation'] + max(realistic_downstream_depth, minimum_depth)
        
        # Calculate energy dissipation with the updated downstream level
        energy_results = calculate_energy_dissipation(
            dam, upstream_level, downstream_level, channel_width
        )
        
        # Check for hydraulic jump
        # Use depth at 20m downstream as reference tailwater depth
        index_20m = min(len(tailwater['x']) - 1, np.argmin(np.abs(tailwater['x'] - 20)))
        tailwater_depth = tailwater['y'][index_20m]
        
        jump = hydraulic_jump_location(
            dam, discharge, channel_width, downstream_slope,
            downstream_channel.roughness, tailwater_depth
        )
        
        # Return all results
        return {
            'upstream_level': upstream_level,
            'discharge': discharge,
            'head': head,
            'velocity': velocity,
            'energy_results': energy_results,
            'tailwater': tailwater,
            'normal_depth': yn,
            'critical_depth': yc,
            'hydraulic_jump': jump
        }
    
    def create_placeholder_results(self, scenario, upstream_level=None):
        """Create simplified placeholder results for testing the UI."""
        # Use provided upstream level or default to flood peak
        if upstream_level is None:
            upstream_level = scenario['flood_water_level']
        
        dam_crest = scenario['dam_crest_elevation']
        dam_base = scenario['dam_base_elevation']
        
        # Check if water level is above crest
        if upstream_level <= dam_crest:
            # No flow over dam
            return {
                'upstream_level': upstream_level,
                'discharge': 0,
                'head': 0,
                'velocity': 0,
                'energy_results': {'energy_loss': 0, 'dissipation_ratio': 0, 'power': 0},
                'tailwater': {'x': np.array([0, 100]), 
                            'z': np.array([dam_base, dam_base - 5]), 
                            'wse': np.array([dam_base, dam_base - 5]), 
                            'y': np.array([0, 0]), 
                            'v': np.array([0, 0]), 
                            'fr': np.array([0, 0])},
                'normal_depth': 0,
                'critical_depth': 0,
                'hydraulic_jump': {'jump_possible': False, 'reason': 'No flow over dam'}
            }
        
        # Simplified calculations for testing
        # Head over crest
        head = upstream_level - dam_crest
        
        # Simplified discharge calculation (approximate ogee weir formula)
        width = scenario['channel_width_at_dam']
        discharge = 1.7 * width * head**(3/2)
        
        # Simplified velocity estimate
        overflow_area = head * width
        velocity = discharge / overflow_area
        
        # Create x-coordinates for the tailwater profile
        x = np.linspace(0, 500, 100)
        
        # Simplified tailwater profile
        # Assume the water level eventually reaches normal depth
        # Approximate normal depth based on channel parameters
        bottom_width = scenario['channel_bottom_width']
        
        # Simple normal depth approximation (Manning's formula)
        roughness = scenario['channel_roughness']
        slope = scenario['downstream_slope']
        
        # Approximate normal and critical depths
        yn = (roughness * discharge / (bottom_width * slope**(1/2)))**(3/5)
        yc = (discharge**2 / (9.81 * bottom_width**2))**(1/3)
        
        # Create bed elevations (sloping downstream)
        z = dam_base - slope * x
        
        # Create water depths (transitioning from initial depth to normal depth)
        # Initial depth after dam (simplified - about 0.7 of critical depth)
        initial_depth = 0.7 * yc
        
        # Create water depth profile transitioning from initial to normal depth
        depth = initial_depth + (yn - initial_depth) * (1 - np.exp(-0.01 * x))
        
        # Create water surface elevations
        wse = z + depth
        
        # Create velocities (Q/A)
        if scenario['channel_type'] == 'rectangular':
            area = bottom_width * depth
        else:  # trapezoidal
            side_slope = scenario['channel_side_slope']
            area = (bottom_width + side_slope * depth) * depth
        
        v = discharge / area
        
        # Create Froude numbers
        fr = v / np.sqrt(9.81 * depth)
        
        # Check for hydraulic jump (simplified)
        jump_possible = np.any(fr > 1) and np.any(fr < 1)
        
        if jump_possible:
            # Find the location of the hydraulic jump (where Fr crosses 1)
            jump_indices = np.where(np.diff(fr > 1))[0]
            if len(jump_indices) > 0:
                jump_index = jump_indices[0]
                jump_loc = x[jump_index]
                
                # Simplified jump characteristics
                fr1 = fr[jump_index]
                y1 = depth[jump_index]
                y2 = 0.5 * y1 * (np.sqrt(1 + 8 * fr1**2) - 1)  # Sequent depth formula
                
                jump = {
                    'jump_possible': True,
                    'location': jump_loc,
                    'initial_depth': y1,
                    'sequent_depth': y2,
                    'initial_froude': fr1,
                    'jump_type': 'Hydraulic Jump',
                    'depth_ratio': y2/y1,
                    'energy_loss': (y2-y1)**3 / (4*y1*y2)  # Simplified energy loss formula
                }
            else:
                jump = {'jump_possible': False, 'reason': 'No transition from supercritical to subcritical'}
        else:
            jump = {'jump_possible': False, 'reason': 'No transition from supercritical to subcritical'}
        
        # Simplified energy dissipation
        upstream_energy = upstream_level + velocity**2 / (2 * 9.81)
        downstream_energy = wse[10] + v[10]**2 / (2 * 9.81)  # Some point downstream
        energy_loss = upstream_energy - downstream_energy
        
        if upstream_level - dam_base > 0:
            dissipation_ratio = energy_loss / (upstream_level - dam_base)
        else:
            dissipation_ratio = 0
        
        energy_results = {
            'upstream_energy': upstream_energy,
            'downstream_energy': downstream_energy,
            'energy_loss': energy_loss,
            'dissipation_ratio': dissipation_ratio,
            'power': 1000 * 9.81 * discharge * energy_loss  # ρ*g*Q*ΔE
        }
        
        # Assemble tailwater dictionary
        tailwater = {
            'x': x,
            'z': z,
            'wse': wse,
            'y': depth,
            'v': v,
            'fr': fr
        }
        
        # Return the complete results dictionary
        return {
            'upstream_level': upstream_level,
            'discharge': discharge,
            'head': head,
            'velocity': velocity,
            'energy_results': energy_results,
            'tailwater': tailwater,
            'normal_depth': yn,
            'critical_depth': yc,
            'hydraulic_jump': jump
        }
    
    def update_profile_view(self):
        """Update the profile visualization."""
        if not self.results:
            return
        
        # Clear the figure
        self.profile_figure.clear()
        ax = self.profile_figure.add_subplot(111)
        
        if USING_EXISTING_MODULES:
            # Use the imported visualization functions
            plot_enhanced_profile(self.scenario, self.results, ax=ax)
        else:
            # Create a basic visualization without external dependencies
            self.create_basic_profile_plot(ax)
        
        self.profile_canvas.draw()
    
    def update_regime_view(self):
        """Update the flow regime visualization."""
        if not self.results:
            return
        
        # Clear the figure
        self.regime_figure.clear()
        
        if USING_EXISTING_MODULES:
            # Use the flow regime dashboard function
            axes = create_flow_regime_dashboard(self.scenario, self.results, figsize=(10, 8))
            # The function should return the figure and axes, but we'll use our own figure
            self.regime_figure = axes[0].figure
            self.regime_canvas.figure = self.regime_figure
        else:
            # Create a basic visualization without external dependencies
            self.create_basic_regime_plot()
        
        self.regime_canvas.draw()
    
    def update_cross_section(self):
        """Update the cross-section visualization."""
        if not self.results:
            return
        
        # Get location
        location = self.cross_location_spin.value()
        
        # Get parameter to highlight
        param_option = self.cross_param_combo.currentText()
        
        # Clear the figure
        self.cross_figure.clear()
        ax = self.cross_figure.add_subplot(111)
        
        # Determine water depth and parameters at this location
        if location <= 0:  # Upstream of dam
            water_elevation = self.results['upstream_level']
            bed_elevation = self.scenario['dam_base_elevation']
            water_depth = max(0, water_elevation - bed_elevation)
            
            # Estimate velocity
            if water_depth > 0 and self.results['discharge'] > 0:
                area = water_depth * self.scenario['channel_width_at_dam']
                velocity = self.results['discharge'] / area
                froude = velocity / np.sqrt(9.81 * water_depth)
            else:
                velocity = 0
                froude = 0
                
            # Estimate shear stress (simplified τ = ρgRS)
            if water_depth > 0:
                rho = 1000  # Water density
                g = 9.81     # Gravity
                R = water_depth  # Simplified hydraulic radius
                S = self.scenario['channel_slope']
                shear = rho * g * R * S
            else:
                shear = 0
                
        else:  # Downstream of dam
            tailwater = self.results['tailwater']
            # Find closest point in tailwater results
            idx = np.argmin(np.abs(tailwater['x'] - location))
            if idx < len(tailwater['wse']):
                water_elevation = tailwater['wse'][idx]
                bed_elevation = tailwater['z'][idx]
                water_depth = max(0, water_elevation - bed_elevation)
                
                # Get velocity and Froude number if available
                velocity = tailwater['v'][idx] if idx < len(tailwater['v']) else 0
                froude = tailwater['fr'][idx] if idx < len(tailwater['fr']) else 0
                
                # Estimate shear stress
                if water_depth > 0:
                    rho = 1000
                    g = 9.81
                    R = water_depth  # Simplified
                    S = self.scenario['downstream_slope']
                    shear = rho * g * R * S
                else:
                    shear = 0
            else:
                water_depth = 0
                velocity = 0
                froude = 0
                shear = 0
        
        # Set up highlight parameter based on selection
        if param_option == "Froude Number" and froude > 0:
            highlight_param = {'type': 'froude', 'value': froude, 'max_value': 2.0}
        elif param_option == "Velocity" and velocity > 0:
            highlight_param = {'type': 'velocity', 'value': velocity, 'max_value': 5.0}
        elif param_option == "Shear Stress" and shear > 0:
            highlight_param = {'type': 'shear', 'value': shear, 'max_value': 100.0}
        else:
            highlight_param = None
        
        if USING_EXISTING_MODULES:
            # Use the imported visualization function
            channel_type = 'rectangular' if self.scenario['channel_type'] == 'rectangular' else 'trapezoidal'
            channel_params = {
                'bottom_width': self.scenario['channel_bottom_width'],
                'side_slope': self.scenario['channel_side_slope']
            }
            
            plot_channel_cross_section(ax, channel_type, channel_params, water_depth, 
                                     highlight_param=highlight_param, annotate=True)
        else:
            # Create a basic visualization without external dependencies
            self.create_basic_cross_section_plot(ax, water_depth, location)
        
        # Set cross-section title
        if location <= 0:
            title = f"Cross-Section at Upstream Location (x={location:.2f}m)"
        elif abs(location) < 0.1:
            title = f"Cross-Section at Dam (x=0m)"
        else:
            title = f"Cross-Section at Downstream Location (x={location:.2f}m)"
            
        # Add key hydraulic parameters to title
        if water_depth > 0:
            title += f"\nDepth={water_depth:.2f}m, V={velocity:.2f}m/s, Fr={froude:.2f}"
        
        ax.set_title(title)
        
        # Make sure the aspect ratio is appropriate
        ax.set_aspect('equal')
        
        self.cross_canvas.draw()
    
    def update_results_text(self):
        """Update the results text display."""
        if not self.results:
            return
        
        # Format the results as text
        text = "<h2>Hydraulic Analysis Results</h2>"
        text += f"<p><b>Upstream Water Level:</b> {self.results['upstream_level']:.2f} m</p>"
        text += f"<p><b>Head Over Crest:</b> {self.results['head']:.2f} m</p>"
        text += f"<p><b>Discharge:</b> {self.results['discharge']:.2f} m³/s</p>"
        text += f"<p><b>Velocity at Crest:</b> {self.results['velocity']:.2f} m/s</p>"
        
        text += "<h3>Downstream Conditions</h3>"
        text += f"<p><b>Normal Depth:</b> {self.results['normal_depth']:.2f} m</p>"
        text += f"<p><b>Critical Depth:</b> {self.results['critical_depth']:.2f} m</p>"
        
        text += "<h3>Energy Dissipation</h3>"
        energy = self.results['energy_results']
        text += f"<p><b>Energy Loss:</b> {energy['energy_loss']:.2f} m</p>"
        text += f"<p><b>Dissipation Ratio:</b> {energy['dissipation_ratio']*100:.1f}%</p>"
        text += f"<p><b>Power Dissipated:</b> {energy['power']/1000:.2f} kW</p>"
        
        text += "<h3>Hydraulic Jump</h3>"
        jump = self.results['hydraulic_jump']
        if jump.get('jump_possible', False):
            text += f"<p><b>Location:</b> {jump['location']:.2f} m downstream</p>"
            text += f"<p><b>Initial Depth:</b> {jump['initial_depth']:.2f} m</p>"
            text += f"<p><b>Sequent Depth:</b> {jump['sequent_depth']:.2f} m</p>"
            text += f"<p><b>Initial Froude Number:</b> {jump['initial_froude']:.2f}</p>"
            text += f"<p><b>Jump Type:</b> {jump['jump_type']}</p>"
            text += f"<p><b>Depth Ratio:</b> {jump['depth_ratio']:.2f}</p>"
            text += f"<p><b>Energy Loss in Jump:</b> {jump.get('energy_loss', 0):.2f} m</p>"
        else:
            reason = jump.get('reason', 'No hydraulic jump detected')
            text += f"<p>{reason}</p>"
        
        # Set the text
        self.results_text.setText(text)
    
    def create_basic_profile_plot(self, ax):
        """Create a basic profile plot without external dependencies."""
        # Extract data
        dam_base = self.scenario['dam_base_elevation']
        dam_crest = self.scenario['dam_crest_elevation']
        upstream_level = self.results['upstream_level']
        tailwater = self.results['tailwater']
        
        # Create x-coordinates
        x_values = tailwater['x']
        
        # Plot channel bed
        ax.plot(x_values, tailwater['z'], 'k-', linewidth=1.5, label='Channel Bed')
        
        # Plot water surface
        ax.plot(x_values, tailwater['wse'], 'b-', linewidth=2, label='Water Surface')
        
        # Fill water area
        ax.fill_between(x_values, tailwater['wse'], tailwater['z'], 
                       color='skyblue', alpha=0.5)
        
        # Draw the dam
        dam_x = [-20, -0.1, 0, 0.1, 5]
        dam_y = [dam_base, dam_base, dam_crest, dam_crest, dam_base]
        ax.plot(dam_x, dam_y, 'k-', linewidth=2)
        ax.fill_between(dam_x, dam_y, dam_base, color='gray')
        
        # Draw upstream water
        ax.plot([-20, 0], [upstream_level, upstream_level], 'b-', linewidth=2)
        ax.fill_between([-20, 0], [upstream_level, upstream_level], 
                       [dam_base, dam_base], color='skyblue', alpha=0.5)
        
        # Add reference lines
        yn = self.results['normal_depth']
        yc = self.results['critical_depth']
        
        if yn > 0:
            ax.axhline(y=tailwater['z'][0] + yn, color='g', linestyle='--', 
                     label=f'Normal Depth ({yn:.2f} m)')
        
        if yc > 0:
            ax.axhline(y=tailwater['z'][0] + yc, color='r', linestyle=':', 
                     label=f'Critical Depth ({yc:.2f} m)')
        
        # Mark hydraulic jump if present
        jump = self.results['hydraulic_jump']
        if jump.get('jump_possible', False):
            ax.axvline(x=jump['location'], color='m', linestyle='-.', 
                     label='Hydraulic Jump')
        
        # Set labels and grid
        ax.set_xlabel('Distance from Dam (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Water Surface Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def create_basic_regime_plot(self):
        """Create a basic flow regime plot without external dependencies."""
        # Create a 2x2 grid for flow regime visualization
        self.regime_figure.clear()
        axs = self.regime_figure.subplots(2, 2)
        
        # Extract data
        tailwater = self.results['tailwater']
        x_values = tailwater['x']
        
        # Plot water surface profile
        self.create_basic_profile_plot(axs[0, 0])
        axs[0, 0].set_title('Water Surface Profile')
        
        # Plot Froude number
        axs[0, 1].plot(x_values, tailwater['fr'], 'r-', linewidth=2)
        axs[0, 1].axhline(y=1, color='k', linestyle='--', 
                       label='Critical Flow (Fr=1)')
        axs[0, 1].set_xlabel('Distance from Dam (m)')
        axs[0, 1].set_ylabel('Froude Number')
        axs[0, 1].set_title('Froude Number Profile')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()
        
        # Plot velocity
        axs[1, 0].plot(x_values, tailwater['v'], 'g-', linewidth=2)
        axs[1, 0].set_xlabel('Distance from Dam (m)')
        axs[1, 0].set_ylabel('Velocity (m/s)')
        axs[1, 0].set_title('Velocity Profile')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot depth
        axs[1, 1].plot(x_values, tailwater['y'], 'b-', linewidth=2)
        axs[1, 1].set_xlabel('Distance from Dam (m)')
        axs[1, 1].set_ylabel('Water Depth (m)')
        axs[1, 1].set_title('Depth Profile')
        axs[1, 1].grid(True, alpha=0.3)
        
        # Add reference lines
        yn = self.results['normal_depth']
        yc = self.results['critical_depth']
        
        if yn > 0:
            axs[1, 1].axhline(y=yn, color='g', linestyle='--', 
                           label=f'Normal Depth ({yn:.2f} m)')
        
        if yc > 0:
            axs[1, 1].axhline(y=yc, color='r', linestyle=':', 
                           label=f'Critical Depth ({yc:.2f} m)')
        
        axs[1, 1].legend()
        
        # Mark hydraulic jump in all plots if present
        jump = self.results['hydraulic_jump']
        if jump.get('jump_possible', False):
            for ax in axs.flatten():
                ax.axvline(x=jump['location'], color='m', linestyle='-.', alpha=0.7)
        
        # Adjust layout
        self.regime_figure.tight_layout()
    
    def create_basic_cross_section_plot(self, ax, water_depth, location):
        """Create a basic cross-section plot without external dependencies."""
        # Get channel parameters
        bottom_width = self.scenario['channel_bottom_width']
        
        if self.scenario['channel_type'] == 'rectangular':
            side_slope = 0
        else:  # trapezoidal
            side_slope = self.scenario['channel_side_slope']
        
        # Calculate dimensions
        half_bottom = bottom_width / 2
        
        if water_depth > 0:
            water_width = bottom_width + 2 * side_slope * water_depth
            half_water = water_width / 2
        else:
            water_width = bottom_width
            half_water = half_bottom
        
        # Draw the channel
        if side_slope > 0:  # Trapezoidal
            # Draw the bed and banks
            ax.plot([-half_bottom, half_bottom], [0, 0], 'k-', linewidth=1.5)
            ax.plot([-half_bottom, -half_water], [0, water_depth], 'k-', linewidth=1.5)
            ax.plot([half_bottom, half_water], [0, water_depth], 'k-', linewidth=1.5)
            
            # Fill the channel bed
            ax.fill_between([-half_bottom-1, half_bottom+1], 0, -0.5, 
                          color='brown', alpha=0.5)
        else:  # Rectangular
            # Draw the bed and banks
            ax.plot([-half_bottom, half_bottom], [0, 0], 'k-', linewidth=1.5)
            ax.plot([-half_bottom, -half_bottom], [0, water_depth+1], 'k-', linewidth=1.5)
            ax.plot([half_bottom, half_bottom], [0, water_depth+1], 'k-', linewidth=1.5)
            
            # Fill the channel bed
            ax.fill_between([-half_bottom, half_bottom], 0, -0.5, 
                          color='brown', alpha=0.5)
        
        # Draw water if present
        if water_depth > 0:
            # Draw water surface
            ax.plot([-half_water, half_water], [water_depth, water_depth], 
                  'b-', linewidth=1.5)
            
            # Fill water area
            if side_slope > 0:  # Trapezoidal
                water_x = [-half_water, -half_bottom, half_bottom, half_water]
                water_y = [water_depth, 0, 0, water_depth]
                ax.fill(water_x, water_y, color='skyblue', alpha=0.5)
            else:  # Rectangular
                ax.fill_between([-half_bottom, half_bottom], 0, water_depth, 
                              color='skyblue', alpha=0.5)
            
            # Add dimension lines and annotations
            # Water depth
            ax.annotate(f"{water_depth:.2f} m", 
                       xy=(half_bottom/2, water_depth/2), 
                       xytext=(half_bottom/2 + 0.5, water_depth/2),
                       ha='left', va='center',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Bottom width
            ax.annotate(f"{bottom_width:.2f} m", 
                       xy=(0, -0.1), 
                       xytext=(0, -0.3),
                       ha='center', va='top',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Water surface width
            if water_width > bottom_width:
                ax.annotate(f"{water_width:.2f} m", 
                           xy=(0, water_depth+0.1), 
                           xytext=(0, water_depth+0.3),
                           ha='center', va='bottom',
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Set equal aspect ratio and grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable limits
        max_width = max(half_water, half_bottom) * 1.5
        max_height = max(water_depth * 1.3, 1)
        ax.set_xlim(-max_width, max_width)
        ax.set_ylim(-0.5, max_height)


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = HydraulicModelingApp()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()