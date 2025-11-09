"""Tests for visualization extensions."""

import unittest
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import imagehash
import io
import os
import tempfile

import mlx.core as mx
from ncps.neurons import CfC
from ncps.wirings import Random, NCP
from ncps.neurons.visualization import WiringVisualizer, PerformanceVisualizer


class TestWiringVisualization(unittest.TestCase):
    """Test wiring visualization extensions."""
    
    def setUp(self):
        """Setup test environment."""
        # Create test model
        self.wiring = NCP(
            inter_neurons=20,
            command_neurons=15,
            motor_neurons=5,
            sensory_fanout=3,
            inter_fanout=3,
            recurrent_command_synapses=5,
            motor_fanin=3
        )
        self.model = CfC(self.wiring)
        
        # Create visualizer
        self.visualizer = WiringVisualizer(self.wiring)
    
    def test_graph_creation(self):
        """Test graph creation."""
        # Check graph properties
        self.assertIsNotNone(self.visualizer.graph)
        self.assertEqual(len(self.visualizer.graph.nodes), self.wiring.units)
        
        # Check edge properties
        adj_matrix = self.wiring.adjacency_matrix
        for i, j in self.visualizer.graph.edges():
            self.assertNotEqual(adj_matrix[i, j], 0)
    
    def test_plot_methods(self):
        """Test plotting methods don't raise errors."""
        try:
            self.visualizer.plot_wiring()
            plt.close()
            
            self.visualizer.plot_connectivity_matrix()
            plt.close()
        except Exception as e:
            self.fail(f"Plotting raised an exception: {e}")
    
    def test_visual_output(self):
        """Test visual output consistency."""
        # Create reference image
        plt.figure()
        self.visualizer.plot_wiring()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Create hash
        img_hash = imagehash.average_hash(Image.open(buf))
        
        # Compare with reference
        self.assertLessEqual(
            img_hash - imagehash.hex_to_hash("reference_hash"),
            5,
            "Visual output differs significantly from reference"
        )


class TestCustomVisualizer(unittest.TestCase):
    """Test custom visualization extensions."""
    
    def setUp(self):
        """Setup test environment."""
        class CustomVisualizer(WiringVisualizer):
            def plot_custom(self):
                plt.figure()
                plt.imshow(self.wiring.adjacency_matrix)
                plt.colorbar()
                return plt.gcf()
        
        self.wiring = Random(units=50, sparsity_level=0.5)
        self.visualizer = CustomVisualizer(self.wiring)
    
    def test_custom_plot(self):
        """Test custom plotting method."""
        fig = self.visualizer.plot_custom()
        self.assertIsNotNone(fig)
        plt.close()
    
    def test_plot_dimensions(self):
        """Test plot dimensions."""
        fig = self.visualizer.plot_custom()
        self.assertEqual(
            fig.get_size_inches().tolist(),
            [6.4, 4.8]  # Default figure size
        )
        plt.close()


class TestInteractiveVisualizer(unittest.TestCase):
    """Test interactive visualization extensions."""
    
    def setUp(self):
        """Setup test environment."""
        class InteractiveVisualizer(WiringVisualizer):
            def create_interactive_plot(self):
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=self.wiring.adjacency_matrix
                ))
                return fig
        
        self.wiring = Random(units=30, sparsity_level=0.5)
        self.visualizer = InteractiveVisualizer(self.wiring)
    
    def test_figure_creation(self):
        """Test Plotly figure creation."""
        fig = self.visualizer.create_interactive_plot()
        self.assertIsInstance(fig, go.Figure)
    
    def test_figure_data(self):
        """Test figure data."""
        fig = self.visualizer.create_interactive_plot()
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, 'heatmap')


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring visualization."""
    
    def setUp(self):
        """Setup test environment."""
        class PerformanceMonitor(PerformanceVisualizer):
            def __init__(self):
                super().__init__()
                self.max_history = 1000
            
            def add_metric(self, name, value):
                if name not in self.history:
                    self.history[name] = []
                if len(self.history[name]) >= self.max_history:
                    self.history[name].pop(0)
                self.history[name].append(value)
        
        self.monitor = PerformanceMonitor()
    
    def test_metric_tracking(self):
        """Test metric tracking."""
        # Add metrics
        for i in range(100):
            self.monitor.add_metric('loss', float(mx.random.uniform().item()))
        
        self.assertEqual(len(self.monitor.history['loss']), 100)
    
    def test_history_limit(self):
        """Test history size limit."""
        # Add more metrics than limit
        for i in range(2000):
            self.monitor.add_metric('metric', i)
        
        self.assertEqual(
            len(self.monitor.history['metric']),
            self.monitor.max_history
        )


class TestVisualizationExport(unittest.TestCase):
    """Test visualization export capabilities."""
    
    def setUp(self):
        """Setup test environment."""
        class ExportableVisualizer(WiringVisualizer):
            def export_plot(self, path, format='png'):
                fig = plt.figure()
                plt.imshow(self.wiring.adjacency_matrix)
                plt.savefig(path, format=format)
                plt.close()
        
        self.wiring = Random(units=20, sparsity_level=0.5)
        self.visualizer = ExportableVisualizer(self.wiring)
        self.temp_dir = tempfile.mkdtemp()
    
    def test_png_export(self):
        """Test PNG export."""
        path = os.path.join(self.temp_dir, 'test.png')
        self.visualizer.export_plot(path)
        self.assertTrue(os.path.exists(path))
        
        # Verify image
        img = Image.open(path)
        self.assertEqual(img.format, 'PNG')
    
    def test_pdf_export(self):
        """Test PDF export."""
        path = os.path.join(self.temp_dir, 'test.pdf')
        self.visualizer.export_plot(path, format='pdf')
        self.assertTrue(os.path.exists(path))
    
    def tearDown(self):
        """Cleanup temporary files."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)


class TestVisualizationStyle(unittest.TestCase):
    """Test visualization style compliance."""
    
    def setUp(self):
        """Setup test environment."""
        class StyledVisualizer(WiringVisualizer):
            def __init__(self, wiring):
                super().__init__(wiring)
                self.style = {
                    'font_size': 12,
                    'line_width': 2,
                    'color_scheme': 'viridis'
                }
            
            def plot_styled(self):
                plt.figure()
                plt.imshow(
                    self.wiring.adjacency_matrix,
                    cmap=self.style['color_scheme']
                )
                plt.tick_params(labelsize=self.style['font_size'])
                return plt.gcf()
        
        self.wiring = Random(units=30, sparsity_level=0.5)
        self.visualizer = StyledVisualizer(self.wiring)
    
    def test_style_properties(self):
        """Test style properties."""
        self.assertEqual(self.visualizer.style['font_size'], 12)
        self.assertEqual(self.visualizer.style['color_scheme'], 'viridis')
    
    def test_styled_plot(self):
        """Test styled plot creation."""
        fig = self.visualizer.plot_styled()
        
        # Check font sizes
        for label in fig.axes[0].get_xticklabels():
            self.assertEqual(label.get_fontsize(), 12)
        
        plt.close()


if __name__ == '__main__':
    unittest.main()
