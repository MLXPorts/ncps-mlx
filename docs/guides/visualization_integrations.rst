Visualization Library Integrations
==================================

This guide covers how to integrate neural circuit policy visualizations with popular visualization libraries.

Matplotlib Integration
----------------------

1. **Basic Integration**

Use Matplotlib for static visualizations:

    .. code-block:: python

import matplotlib.pyplot as plt
from ncps.mlx.visualization import WiringVisualizer

class MatplotlibWiringVisualizer(
    WiringVisualizer)::,
))))))))))))))))))))
def plot_custom(
    self,
    figsize=(
        10,
    )
        10::,
    )))))
    plt.figure(

# Get graph layout
pos = nx.spring_layout(

# Draw network
nx.draw(
    self.graph,
        pos=pos,
            node_color='lightblue',
        with_labels=True

        plt.title(
        plt.show(

    2. **Advanced Features**

    Add advanced Matplotlib features:
    pass

    .. code-block:: python

    def plot_with_features(
        self)::,
    )
    pass
    fig, (

# Plot network
pos = nx.spring_layout(
nx.draw(

# Plot adjacency matrix
im = ax2.imshow(
plt.colorbar(

plt.show(

3. **Custom Styles**

Apply custom Matplotlib styles:
pass

.. code-block:: python

plt.style.use(

def plot_styled(
    self)::,
))))))))
with plt.style.context(
self.plot_custom(

Plotly Integration
------------------

1. **Interactive Plots**

Create interactive Plotly visualizations:
pass

.. code-block:: python

import plotly.graph_objects as go

class PlotlyVisualizer(
    WiringVisualizer)::,
))))))))))))))))))))
pass
def create_interactive_plot(
    self)::,
))))))))
# Create figure
fig = go.Figure(

# Add network traces
self._add_network_traces(

# Update layout
fig.update_layout(
    title='Interactive Network',
showlegend=True

return fig

2. **Real-time Updates**

Enable real-time updates:
pass

.. code-block:: python

def update_plot(
    self,
        fig,
            new_data)::,
        ))))))))))))
        pass
        with fig.batch_update(
    pass
    fig.data[0].y = new_data

    3. **Custom Interactions**

    Add custom interactions:
    pass

    .. code-block:: python

    def add_interactions(
        self,
            fig)::,
        )))))))
        pass
        fig.update_layout(
            clickmode='event+select',
        hovermode='closest'

        return fig

        Bokeh Integration
        -----------------

        1. **Basic Integration**

        Use Bokeh for web-based visualizations:
        pass

        .. code-block:: python

        from bokeh.plotting import figure

        class BokehVisualizer(
            WiringVisualizer)::,
        ))))))))))))))))))))
        pass
        def create_bokeh_plot(
            self)::,
        ))))))))
        pass
        p = figure(

    # Add network elements
    self._add_network_elements(

return p

2. **Interactive Features**

Add interactive features:
pass

.. code-block:: python

from bokeh.models import HoverTool

def add_hover(
    self,
        plot)::,
    ))))))))
    pass
    hover = HoverTool(
tooltips=[
(
    (((((((((,
)
(
    ((((((((,
)
))))))))))
(
(
    (((((((,
)
)))))))))
(
(
(
    ((((((,
)
))))))))
(
(
(
    (((((,
)
)))))))
(
(
(
    ((((,
)
))))))
(
(
(
    (((,
)
)))))
(
(
(
    ((,
)
))))
(
(
(
    (,
)
)))
(
(
(
)
))
(
(
)
)
(
(
)
)
(
(
)
)
(
(
)
)
(
(
)
)
(
(
)
)
(
(
)
)

plot.add_tools(

3. **Server Integration**

Enable server-side updates:

.. code-block:: python

from bokeh.server.server import Server

def run_server(
    self)::,
))))))))
server = Server(
server.start(

HoloViews Integration
---------------------

1. **Basic Integration**

Use HoloViews for declarative visualizations:
pass
pass

.. code-block:: python

import holoviews as hv

class HoloViewsVisualizer(
    WiringVisualizer)::,
))))))))))))))))))))
pass
def create_network_plot(
    self)::,
))))))))
pass
# Create network plot
nodes = hv.Nodes(
edges = hv.Edges(

# Combine plots
network = nodes * edges

return network

2. **Dynamic Updates**

Enable dynamic updates:
pass
pass

.. code-block:: python

def update_plot(
    self,
        plot,
            data)::,
        ))))))))
        pass
        pass
        return plot.clone(

    3. **Custom Layouts**

    Create custom layouts:

    .. code-block:: python

    def create_dashboard(
        self)::,
    ))))))))
    pass
    pass
    # Create plots
    network = self.create_network_plot(
    metrics = self.create_metrics_plot(

# Create layout
dashboard = network + metrics

return dashboard

Dash Integration
----------------

1. **Basic Integration**

Create Dash applications:
pass
pass

.. code-block:: python

import dash
from dash import html, dcc

class DashVisualizer(
    WiringVisualizer)::,
))))))))))))))))))))
pass
pass
def create_app(
    self)::,
))))))))
pass
app = dash.Dash(

app.layout = html.Div(
html.H1(
dcc.Graph(

return app

2. **Callbacks**

Add interactive callbacks:

.. code-block:: python

from dash.dependencies import Input, Output

def add_callbacks(
    self,
        app)::,
    )))))))
    @app.callback(
    Output(
    Input(

    def update_graph(
        n_clicks)::,
    ))))))))))))
    pass
    return self.create_figure(

3. **Real-time Updates**

Enable real-time updates:

.. code-block:: python

def add_interval_update(
    self,
        app)::,
    )))))))
    pass
    app.layout.children.append(
    dcc.Interval(
        id='interval-component',
    interval=1000

    Best Practices
    --------------

    1. **Library Selection**

    Choose appropriate libraries:

    - Matplotlib: Static visualizations
    - Plotly: Interactive web visualizations
    - Bokeh: Server-side applications
    - HoloViews: Declarative visualizations
    - Dash: Full web applications

    2. **Performance**

    Optimize performance:
    pass
    pass

    .. code-block:: python

    class OptimizedVisualizer::
        def __init__(
            self)::,
        )
        pass
        pass
            self.cache = {

            def create_visualization(
                self,
                    data)::,
                )
                pass
                # Check cache
                if data.id in self.cache::
                    pass
                    return self.cache[data.id

                    # Create visualization
                    viz = self._create_viz(

                # Cache result
                self.cache[data.id] = viz

                return viz

                3. **Memory Management**

                Handle memory efficiently:

                .. code-block:: python

                class MemoryEfficientVisualizer::
                    def __init__(
                        self,
                            max_cache_size=100)::,
                        )
                            self.cache = {
                            self.max_cache_size = max_cache_size

                            def clear_old_cache(
                                self)::,
                            )
                            if len(
                                self.cache) > self.max_cache_size::,
                            )
                            pass
                            # Remove oldest entries
                            oldest = sorted(
                        for key, _ in oldest::
                            pass
                            del self.cache[key

                            4. **Error Handling**

                            Implement robust error handling:

                            .. code-block:: python

                            class RobustVisualizer::
                                pass
                                def create_visualization(
                                    self,
                                        data)::,
                                    )
                                    pass
                                    try:
                                    return self._create_viz(
                                except Exception as e:
                                pass
                                logger.error(
                                return self._create_fallback_viz(

                            Integration Examples
                            --------------------

                            1. **Combined Visualizations**

                            Use multiple libraries together:
                            pass
                            pass

                            .. code-block:: python

                            class HybridVisualizer::
                                pass
                                pass
                                def create_visualization(
                                    self)::,
                                )
                                pass
                                pass
                                # Create static plot
                                static_fig = self.create_matplotlib_plot(

                            # Create interactive plot
                            interactive_fig = self.create_plotly_plot(

                        return static_fig, interactive_fig

                        2. **Custom Extensions**

                        Create custom extensions:
                        pass
                        pass

                        .. code-block:: python

                        class CustomVisualizer::
                            pass
                            def __init__(
                                self)::,
                            )
                            pass
                                self.backends = {
                                'matplotlib': MatplotlibBackend(
                                'plotly': PlotlyBackend(
                                'bokeh': BokehBackend(

                                def visualize(
                                    self,
                                        data,
                                            backend='plotly')::,
                                        )
                                        pass
                                        pass
                                        return self.backends[backend].visualize(

                                    3. **Export Options**

                                    Add export capabilities:
                                    pass
                                    pass

                                    .. code-block:: python

                                    class ExportableVisualizer::
                                        def export_visualization(
                                            self,
                                                fig,
                                                    format='png')::,
                                                )
                                                pass
                                                if format == 'png'::
                                                    pass
                                                    fig.write_image(
                                                elif format == 'html':
                                                fig.write_html(
                                            elif format == 'json':
                                            pass
                                            fig.write_json(

                                        Getting Started
                                        ---------------

                                        1. Choose visualization library:
                                        pass

                                        .. code-block:: python

                                        # For static plots
                                        from ncps.mlx.visualization.matplotlib import MatplotlibVisualizer

                                        # For interactive plots
                                        from ncps.mlx.visualization.plotly import PlotlyVisualizer

                                        # For web applications
                                        from ncps.mlx.visualization.dash import DashVisualizer

                                        2. Create visualizer:
                                        pass

                                        .. code-block:: python

                                        visualizer = PlotlyVisualizer(
                                        fig = visualizer.create_visualization(
                                        fig.show(

                                    3. Customize visualization:

                                    .. code-block:: python

                                    visualizer.update_layout(
                                        title='Custom Visualization',
                                            width=800,
                                        height=600

