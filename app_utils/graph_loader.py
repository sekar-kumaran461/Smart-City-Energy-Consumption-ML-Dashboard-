"""
Graph Loader Utility for Streamlit Application
Provides easy access to all generated graphs organized by analysis type
"""

from pathlib import Path
from typing import Dict, List
import streamlit as st


class GraphLoader:
    """Utility class to load and organize generated analysis graphs."""
    
    def __init__(self, base_path: str = "generated_graphs"):
        """Initialize the graph loader with the base path to graphs."""
        self.base_path = Path(base_path)
        self.univariate_path = self.base_path / "univariate"
        self.bivariate_path = self.base_path / "bivariate"
        self.multivariate_path = self.base_path / "multivariate"
        self.advanced_path = self.base_path / "advanced"
        
    def get_graph_path(self, category: str, graph_number: int) -> Path:
        """
        Get the path to a specific graph.
        
        Args:
            category: One of 'univariate', 'bivariate', 'multivariate', 'advanced'
            graph_number: The graph number (1-27)
            
        Returns:
            Path object to the graph file
        """
        category_map = {
            'univariate': self.univariate_path,
            'bivariate': self.bivariate_path,
            'multivariate': self.multivariate_path,
            'advanced': self.advanced_path
        }
        
        if category not in category_map:
            raise ValueError(f"Invalid category: {category}")
        
        folder = category_map[category]
        # Find the graph file with the matching number
        graphs = sorted(folder.glob(f"{graph_number:02d}_*.png"))
        
        if graphs:
            return graphs[0]
        return None
    
    def get_all_graphs(self, category: str = None) -> Dict[str, List[Path]]:
        """
        Get all graphs, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary mapping category names to lists of graph paths
        """
        if category:
            folder_map = {
                'univariate': self.univariate_path,
                'bivariate': self.bivariate_path,
                'multivariate': self.multivariate_path,
                'advanced': self.advanced_path
            }
            folder = folder_map.get(category)
            if folder:
                return {category: sorted(folder.glob("*.png"))}
            return {}
        
        return {
            'univariate': sorted(self.univariate_path.glob("*.png")),
            'bivariate': sorted(self.bivariate_path.glob("*.png")),
            'multivariate': sorted(self.multivariate_path.glob("*.png")),
            'advanced': sorted(self.advanced_path.glob("*.png"))
        }
    
    def display_graph(self, category: str, graph_number: int, caption: str = None, use_column_width: bool = True):
        """
        Display a graph in Streamlit.
        
        Args:
            category: Graph category
            graph_number: Graph number
            caption: Optional caption for the graph
            use_column_width: Use full column width (compatibility parameter)
        """
        graph_path = self.get_graph_path(category, graph_number)
        if graph_path and graph_path.exists():
            st.image(str(graph_path), caption=caption, use_column_width=use_column_width)
        else:
            st.error(f"Graph not found: {category} #{graph_number}")
    
    def display_category(self, category: str, columns: int = 2):
        """
        Display all graphs from a category in a grid layout.
        
        Args:
            category: The category to display
            columns: Number of columns in the grid (default: 2)
        """
        graphs = self.get_all_graphs(category).get(category, [])
        
        if not graphs:
            st.warning(f"No graphs found for category: {category}")
            return
        
        # Create grid layout
        for i in range(0, len(graphs), columns):
            cols = st.columns(columns)
            for j, col in enumerate(cols):
                if i + j < len(graphs):
                    graph_path = graphs[i + j]
                    with col:
                        # Extract title from filename
                        title = graph_path.stem.replace('_', ' ').title()
                        st.image(str(graph_path), caption=title, use_column_width=True)
    
    def get_graph_info(self) -> Dict:
        """
        Get information about all available graphs.
        
        Returns:
            Dictionary with graph statistics
        """
        all_graphs = self.get_all_graphs()
        return {
            'total_graphs': sum(len(graphs) for graphs in all_graphs.values()),
            'by_category': {cat: len(graphs) for cat, graphs in all_graphs.items()},
            'categories': list(all_graphs.keys())
        }


# Graph metadata for easy reference
GRAPH_METADATA = {
    'univariate': {
        1: {'name': 'Electricity Load Distribution', 'description': 'Distribution and boxplot of electricity load'},
        2: {'name': 'Temperature Distribution', 'description': 'Temperature histogram and density plot'},
        3: {'name': 'Renewable Energy Distribution', 'description': 'Solar, wind, renewable, and net load distributions'},
        4: {'name': 'Temporal Patterns', 'description': 'Distribution by hour, day of week, and season'},
        5: {'name': 'Battery and Grid Status', 'description': 'Battery SOC, grid frequency, power factor, and voltage'},
        6: {'name': 'Weather Conditions', 'description': 'Humidity, solar irradiance, wind speed, and cloud cover'},
        7: {'name': 'EV and Transit Load', 'description': 'EV charging, transit load, and mobility score'},
        8: {'name': 'Categorical Variables', 'description': 'Weekend, peak load, demand response, and curtailment'},
    },
    'bivariate': {
        9: {'name': 'Load vs Temperature', 'description': 'Relationship between electricity load and temperature'},
        10: {'name': 'Load vs Time', 'description': 'Load patterns by hour and day of week'},
        11: {'name': 'Renewable vs Weather', 'description': 'Solar and wind output vs weather conditions'},
        12: {'name': 'Load by Season and Time', 'description': 'Seasonal load patterns throughout the day'},
        13: {'name': 'Battery vs Load', 'description': 'Battery SOC and discharge rate vs load'},
        14: {'name': 'EV and Mobility', 'description': 'EV charging vs mobility and traffic patterns'},
        15: {'name': 'Load vs Building Occupancy', 'description': 'Building energy consumption patterns'},
        16: {'name': 'Power Factor vs Voltage', 'description': 'Grid electrical characteristics relationships'},
    },
    'multivariate': {
        17: {'name': 'Energy Correlation Heatmap', 'description': 'Correlations between energy variables'},
        18: {'name': 'Weather Correlation Heatmap', 'description': 'Correlations between weather variables'},
        19: {'name': '3D Load Analysis', 'description': '3D visualization of load vs temperature vs humidity'},
        20: {'name': 'Pair Plot Analysis', 'description': 'Pairwise relationships between key variables'},
        21: {'name': 'Multi-series Time Pattern', 'description': 'Multiple variables over time comparison'},
        22: {'name': 'Grouped Analysis', 'description': 'Load analysis by season, weekend, and area type'},
    },
    'advanced': {
        23: {'name': 'Renewable Energy Mix', 'description': 'Renewable energy composition and penetration analysis'},
        24: {'name': 'Grid Stability Analysis', 'description': 'Frequency, voltage, and power factor stability'},
        25: {'name': 'Demand Response Analysis', 'description': 'DR events and curtailment patterns'},
        26: {'name': 'Urban Mobility Analysis', 'description': 'EV charging and transit energy nexus'},
        27: {'name': 'Peak Load Analysis', 'description': 'Peak load factors and prediction patterns'},
    }
}


def get_graph_description(graph_number: int) -> Dict:
    """
    Get metadata for a specific graph by number.
    
    Args:
        graph_number: The graph number (1-27)
        
    Returns:
        Dictionary with graph metadata
    """
    for category, graphs in GRAPH_METADATA.items():
        if graph_number in graphs:
            return {
                'category': category,
                **graphs[graph_number]
            }
    return None
