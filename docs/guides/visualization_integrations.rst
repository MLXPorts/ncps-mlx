Visualization Library Integrations
==================================
==================================
==================================
==================================
==================================
==================================
==================================
==================================
==================================
==================================
==================================
==================================
==================================
==================================
==================================
===========================

This guide covers how to integrate neural circuit policy visualizations with popular visualization libraries.

Matplotlib Integration
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
------------------

1. **Basic Integration**
   
   Use Matplotlib for static visualizations:

    .. code-block:: python

       import matplotlib.pyplot as plt
       from ncps.mlx.visualization import WiringVisualizer
       
       class MatplotlibWiringVisualizer(WiringVisualizer):
           def plot_custom(self, figsize=(10, 10)):
               plt.figure(figsize=figsize)
               
               # Get graph layout
               pos = nx.spring_layout(self.graph)
               
               # Draw network
               nx.draw(
                   self.graph,
                   pos=pos,
                   node_color='lightblue',
                   with_labels=True
               )
               
               plt.title('Network Visualization')
               plt.show()

2. **Advanced Features**
   
   Add advanced Matplotlib features:

    .. code-block:: python

       def plot_with_features(self):
           fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
           
           # Plot network
           pos = nx.spring_layout(self.graph)
           nx.draw(self.graph, pos=pos, ax=ax1)
           
           # Plot adjacency matrix
           im = ax2.imshow(self.wiring.adjacency_matrix)
           plt.colorbar(im, ax=ax2)
           
           plt.show()

3. **Custom Styles**
   
   Apply custom Matplotlib styles:

    .. code-block:: python

       plt.style.use('seaborn')
       
       def plot_styled(self):
           with plt.style.context('dark_background'):
               self.plot_custom()

Plotly Integration
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
------------------
--------------

1. **Interactive Plots**
   
   Create interactive Plotly visualizations:

    .. code-block:: python

       import plotly.graph_objects as go
       
       class PlotlyVisualizer(WiringVisualizer):
           def create_interactive_plot(self):
               # Create figure
               fig = go.Figure()
               
               # Add network traces
               self._add_network_traces(fig)
               
               # Update layout
               fig.update_layout(
                   title='Interactive Network',
                   showlegend=True
               )
               
               return fig

2. **Real-time Updates**
   
   Enable real-time updates:

    .. code-block:: python

       def update_plot(self, fig, new_data):
           with fig.batch_update():
               fig.data[0].y = new_data

3. **Custom Interactions**
   
   Add custom interactions:

    .. code-block:: python

       def add_interactions(self, fig):
           fig.update_layout(
               clickmode='event+select',
               hovermode='closest'
           )
           
           return fig

Bokeh Integration
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-------------

1. **Basic Integration**
   
   Use Bokeh for web-based visualizations:

    .. code-block:: python

       from bokeh.plotting import figure
       
       class BokehVisualizer(WiringVisualizer):
           def create_bokeh_plot(self):
               p = figure(title='Network Visualization')
               
               # Add network elements
               self._add_network_elements(p)
               
               return p

2. **Interactive Features**
   
   Add interactive features:

    .. code-block:: python

       from bokeh.models import HoverTool
       
       def add_hover(self, plot):
           hover = HoverTool(
               tooltips=[
                   ('Node', '@index'),
                   ('Value', '@value')
               ]
           )
           plot.add_tools(hover)

3. **Server Integration**
   
   Enable server-side updates:

    .. code-block:: python

       from bokeh.server.server import Server
       
       def run_server(self):
           server = Server({'/': self.create_app})
           server.start()

HoloViews Integration
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
-----------------

1. **Basic Integration**
   
   Use HoloViews for declarative visualizations:

    .. code-block:: python

       import holoviews as hv
       
       class HoloViewsVisualizer(WiringVisualizer):
           def create_network_plot(self):
               # Create network plot
               nodes = hv.Nodes(self.graph)
               edges = hv.Edges(self.graph)
               
               # Combine plots
               network = nodes * edges
               
               return network

2. **Dynamic Updates**
   
   Enable dynamic updates:

    .. code-block:: python

       def update_plot(self, plot, data):
           return plot.clone(data)

3. **Custom Layouts**
   
   Create custom layouts:

    .. code-block:: python

       def create_dashboard(self):
           # Create plots
           network = self.create_network_plot()
           metrics = self.create_metrics_plot()
           
           # Create layout
           dashboard = network + metrics
           
           return dashboard

Dash Integration
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
------------

1. **Basic Integration**
   
   Create Dash applications:

    .. code-block:: python

       import dash
       from dash import html, dcc
       
       class DashVisualizer(WiringVisualizer):
           def create_app(self):
               app = dash.Dash(__name__)
               
               app.layout = html.Div([
                   html.H1('Network Visualization'),
                   dcc.Graph(id='network-graph')
               ])
               
               return app

2. **Callbacks**
   
   Add interactive callbacks:

    .. code-block:: python

       from dash.dependencies import Input, Output
       
       def add_callbacks(self, app):
           @app.callback(
               Output('network-graph', 'figure'),
               Input('update-button', 'n_clicks')
           )
           def update_graph(n_clicks):
               return self.create_figure()

3. **Real-time Updates**
   
   Enable real-time updates:

    .. code-block:: python

       def add_interval_update(self, app):
           app.layout.children.append(
               dcc.Interval(
                   id='interval-component',
                   interval=1000
               )
           )

Best Practices
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
-----------

1. **Library Selection**
   
   Choose appropriate libraries:

    - Matplotlib: Static visualizations
    - Plotly: Interactive web visualizations
    - Bokeh: Server-side applications
    - HoloViews: Declarative visualizations
    - Dash: Full web applications

2. **Performance**
   
   Optimize performance:

    .. code-block:: python

       class OptimizedVisualizer:
           def __init__(self):
               self.cache = {}
           
           def create_visualization(self, data):
               # Check cache
               if data.id in self.cache:
                   return self.cache[data.id]
               
               # Create visualization
               viz = self._create_viz(data)
               
               # Cache result
               self.cache[data.id] = viz
               
               return viz

3. **Memory Management**
   
   Handle memory efficiently:

    .. code-block:: python

       class MemoryEfficientVisualizer:
           def __init__(self, max_cache_size=100):
               self.cache = {}
               self.max_cache_size = max_cache_size
           
           def clear_old_cache(self):
               if len(self.cache) > self.max_cache_size:
                   # Remove oldest entries
                   oldest = sorted(self.cache.items())[:-self.max_cache_size]
                   for key, _ in oldest:
                       del self.cache[key]

4. **Error Handling**
   
   Implement robust error handling:

    .. code-block:: python

       class RobustVisualizer:
           def create_visualization(self, data):
               try:
                   return self._create_viz(data)
               except Exception as e:
                   logger.error(f"Visualization error: {e}")
                   return self._create_fallback_viz()

Integration Examples
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
----------------

1. **Combined Visualizations**
   
   Use multiple libraries together:

    .. code-block:: python

       class HybridVisualizer:
           def create_visualization(self):
               # Create static plot
               static_fig = self.create_matplotlib_plot()
               
               # Create interactive plot
               interactive_fig = self.create_plotly_plot()
               
               return static_fig, interactive_fig

2. **Custom Extensions**
   
   Create custom extensions:

    .. code-block:: python

       class CustomVisualizer:
           def __init__(self):
               self.backends = {
                   'matplotlib': MatplotlibBackend(),
                   'plotly': PlotlyBackend(),
                   'bokeh': BokehBackend()
               }
           
           def visualize(self, data, backend='plotly'):
               return self.backends[backend].visualize(data)

3. **Export Options**
   
   Add export capabilities:

    .. code-block:: python

       class ExportableVisualizer:
           def export_visualization(self, fig, format='png'):
               if format == 'png':
                   fig.write_image('visualization.png')
               elif format == 'html':
                   fig.write_html('visualization.html')
               elif format == 'json':
                   fig.write_json('visualization.json')

Getting Started
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
------------

1. Choose visualization library:

    .. code-block:: python

       # For static plots
       from ncps.mlx.visualization.matplotlib import MatplotlibVisualizer
       
       # For interactive plots
       from ncps.mlx.visualization.plotly import PlotlyVisualizer
       
       # For web applications
       from ncps.mlx.visualization.dash import DashVisualizer

2. Create visualizer:

    .. code-block:: python

       visualizer = PlotlyVisualizer(model)
       fig = visualizer.create_visualization()
       fig.show()

3. Customize visualization:

    .. code-block:: python

       visualizer.update_layout(
           title='Custom Visualization',
           width=800,
           height=600
       )
