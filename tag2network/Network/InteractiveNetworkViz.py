# -*- coding: utf-8 -*-

import math
import networkx as nx

from bokeh.io import show, output_file, output_notebook
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, StaticLayoutProvider, Circle, MultiLine
from bokeh.models import HoverTool, TapTool, BoxSelectTool
from bokeh.models import NodesAndLinkedEdges
from bokeh.core.properties import field

#from bokeh.palettes import Spectral8

import Network.DrawNetwork as dn

# draw network using categorical attribute for node colors
# draw using Bokeh
# nodes (df) must include x,y coordinates
# links are specified either in networkx object (nw) or list of links in a DataFrame
#
def drawInteractiveNW(df, nw=None, edgesdf=None, color_attr="Cluster",
                      label_attr="name", title="Interactive Network Visualization",
                      plotfile=None, inline=False):
    def buildNetworkX(linksdf, id1='Source', id2='Target', directed=False):
        linkdata = [(getattr(link, id1), getattr(link, id2)) for link in linksdf.itertuples()]
        g = nx.DiGraph() if directed else nx.Graph()
        g.add_edges_from(linkdata)
        return g

    if inline:
        output_notebook()
    if plotfile:
        output_file(plotfile+".html")
    if nw is None:
        if edgesdf is None:
            print("Must specify either network or edges DataFrame")
            return
        nw = buildNetworkX(edgesdf)
    node_colors, edge_colors, attr_colors = dn.getCategoricalColors(nw, df, color_attr, None, True, None)
    xmin = df['x'].min()
    xmax = df['x'].max()
    ymin = df['y'].min()
    ymax = df['y'].max()
    rng = max((xmax-xmin), (ymax-ymin))
    nNodes = len(df)
    size = 4*rng/math.sqrt(nNodes)
    node_indices = list(range(nNodes))

    tooltips=[
        (label_attr, "@"+label_attr),
        (color_attr, "@"+color_attr)
    ]

    plot = figure(title=title, plot_width=800, plot_height=800, x_range=(xmin-size, xmax+size),
                  y_range=(ymin-size, ymax+size),
                  tools="pan,wheel_zoom,box_zoom,reset", toolbar_location="right", output_backend="webgl")
    plot.add_tools(HoverTool(tooltips=tooltips), TapTool(), BoxSelectTool())

    graph = GraphRenderer()

    # set node renderer and data
    graph.node_renderer.glyph = Circle(size=size, fill_color="fill_color", line_color='gray')
    graph.node_renderer.selection_glyph = Circle(size=size, fill_color="fill_color", line_color='black', line_width=2)
    graph.node_renderer.hover_glyph = Circle(size=size, fill_color="fill_color", line_color='black')
    graph.node_renderer.data_source.data = {'index': node_indices,
                                            label_attr: df[label_attr].tolist(),
                                            'fill_color': node_colors,
                                            color_attr: df[color_attr].fillna('').tolist(),
                                            'x': df['x'].tolist(),
                                            'y': df['y'].tolist(),
                                            }
    # set edge renderer and data
    graph.edge_renderer.glyph = MultiLine(line_color="line_color", line_width=1)
    graph.edge_renderer.data_source.data = {'start': [e[0] for e in nw.edges],
                                            'end': [e[1] for e in nw.edges],
                                            'line_color': edge_colors
                                            }
    # set layout
    graph_layout = dict(zip(node_indices, zip(df['x'], df['y'])))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # set legend by adding dummy glyph with legend
    plot.circle( x='x', y='y', radius=0.0, color='fill_color', legend=field(color_attr), source=graph.node_renderer.data_source.data)
    plot.legend.location = "top_left"
    plot.legend.padding = 0
    plot.legend.margin = 0

    graph.selection_policy = NodesAndLinkedEdges()

    # hide axes and grids
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.xgrid.visible = False
    plot.ygrid.visible = False

    plot.renderers.append(graph)

    show(plot)
