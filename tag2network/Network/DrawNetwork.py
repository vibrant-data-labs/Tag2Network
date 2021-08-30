# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:41:04 2016

@author: rich williams

Modified and extended networkx drawing code

#    Copyright (C) 2004-2016 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#
# Author: Aric Hagberg (hagberg@lanl.gov)

**********
Matplotlib
**********
Draw networks with matplotlib.
See Also
--------
matplotlib:     http://matplotlib.org/
pygraphviz:     http://pygraphviz.github.io/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
import math
import numbers
import numpy as np
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx_labels

__all__ = ['_draw_networkx_']


def _draw_networkx_(G, pos=None, with_labels=True, draw_edges=True, **kwds):
    """Draw the graph G using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    G : graph
       A networkx graph
    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See networkx.layout for functions that compute node positions.
    with_labels :  bool, optional (default=True)
       Set to True to draw labels on the nodes.
    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.
    nodelist : list, optional (default G.nodes())
       Draw only specified nodes
    edgelist : list, optional (default=G.edges())
       Draw only specified edges
    node_size : scalar or array, optional (default=300)
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.
    node_color : color string, or array of floats, (default='r')
       Node color. Can be a single color format string,
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.
    node_shape :  string, optional (default='o')
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8'.
    highlight : array of boolean, if true a black ring is draw round the node
    alpha : float, optional (default=1.0)
       The node and edge transparency
    cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of nodes
    vmin,vmax : float, optional (default=None)
       Minimum and maximum for node colormap scaling
    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)
    width : float, optional (default=1.0)
       Line width of edges
    edge_color : color string, or array of floats (default='r')
       Edge color. Can be a single color format string,
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.
    edge_cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of edges
    edge_vmin,edge_vmax : floats, optional (default=None)
       Minimum and maximum for edge colormap scaling
    style : string, optional (default='solid')
       Edge line style (solid|dashed|dotted,dashdot)
    labels : dictionary, optional (default=None)
       Node labels in a dictionary keyed by node of text labels
    font_size : int, optional (default=12)
       Font size for text labels
    font_color : string, optional (default='k' black)
       Font color string
    font_weight : string, optional (default='normal')
       Font weight
    font_family : string, optional (default='sans-serif')
       Font family
    label : string, optional
        Label for graph legend
    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G,pos=nx.spring_layout(G)) # use spring layout
    >>> import matplotlib.pyplot as plt
    >>> limits=plt.axis('off') # turn of axis
    Also see the NetworkX drawing examples at
    http://networkx.github.io/documentation/latest/gallery.html
    See Also
    --------
    draw()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    if pos is None:
        pos = nx.drawing.spring_layout(G)  # default to spring layout

    draw_nx_nodes(G, pos, **kwds)
    if draw_edges:
        draw_nx_tapered_edges(G, pos, **kwds)
    if with_labels:
        draw_networkx_labels(G, pos, **kwds)
    plt.draw_if_interactive()


def draw_nx_tapered_edges(G, pos,
                          edgelist=None,
                          width=0.5,
                          edge_color='k',
                          style='solid',
                          alpha=1.0,
                          edge_cmap=None,
                          edge_vmin=None,
                          edge_vmax=None,
                          ax=None,
                          label=None,
                          highlight=None,
                          tapered=False,
                          **kwds):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph
    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.
    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())
    width : float, or array of floats
       Line width of edges (default=1.0)
    edge_color : color string, or array of floats
       Edge color. Can be a single color format string (default='r'),
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.
    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)
    alpha : float
       The edge transparency (default=1.0)
    edge_ cmap : Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)
    edge_vmin,edge_vmax : floats
       Minimum and maximum for edge colormap scaling (default=None)
    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.
    label : [None| string]
       Label for legend
    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges
    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> edges=nx.draw_networkx_edges(G,pos=nx.spring_layout(G))
    Also see the NetworkX drawing examples at
    http://networkx.github.io/documentation/latest/gallery.html
    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    if highlight is not None and (isinstance(edge_color, str) or not cb.iterable(edge_color)):
        idMap = {}
        nodes = G.nodes()
        for i in range(len(nodes)):
            idMap[nodes[i]] = i
        ecol = [edge_color]*len(edgelist)
        eHighlight = [highlight[idMap[edge[0]]] or highlight[idMap[edge[1]]] for edge in edgelist]
        for i in range(len(eHighlight)):
            if eHighlight[i]:
                ecol[i] = '0.0'
        edge_color = ecol

    # set edge positions
    if not np.iterable(width):
        lw = np.full(len(edgelist), width)
    else:
        lw = width

    edge_pos = []
    wdScale = 0.01
    for i in range(len(edgelist)):
        e = edgelist[i]
        w = wdScale*lw[i]/2
        p0 = pos[e[0]]
        p1 = pos[e[1]]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        l = math.sqrt(dx*dx+dy*dy)
        edge_pos.append(((p0[0]+w*dy/l, p0[1]-w*dx/l), (p0[0]-w*dy/l, p0[1]+w*dx/l), (p1[0], p1[1])))

    edge_vertices = np.asarray(edge_pos)

    if not isinstance(edge_color, str) \
           and np.iterable(edge_color) \
           and len(edge_color) == len(edge_vertices):
        if np.alltrue([isinstance(c, str)
                         for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                 for c in edge_color])
        elif np.alltrue([not isinstance(c, str)
                           for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if np.alltrue([cb.iterable(c) and len(c) in (3, 4)
                             for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must consist of either color names or numbers')
    else:
        if isinstance(edge_color, str) or len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color, alpha), )
        else:
            raise ValueError('edge_color must be a single color or list of exactly m colors where m is the number or edges')

    if tapered:
        edge_collection = PolyCollection(edge_vertices,
                                         facecolors=edge_colors,
                                         linewidths=0,
                                         antialiaseds=(1,),
                                         transOffset = ax.transData,
                                         )
    else:
        edge_collection = LineCollection(edge_pos,
                                         colors=edge_colors,
                                         linewidths=lw,
                                         antialiaseds=(1,),
                                         linestyle=style,
                                         transOffset = ax.transData,
                                         )

    edge_collection.set_zorder(1)  # edges go behind nodes
    edge_collection.set_label(label)
    ax.add_collection(edge_collection)

    # Set alpha globally if provided as a scalar.
    if isinstance(alpha, numbers.Number):
        edge_collection.set_alpha(alpha)
    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, Colormap))
        edge_collection.set_array(np.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()

    # update view
    minx = np.amin(np.ravel(edge_vertices[:, :, 0]))
    maxx = np.amax(np.ravel(edge_vertices[:, :, 0]))
    miny = np.amin(np.ravel(edge_vertices[:, :, 1]))
    maxy = np.amax(np.ravel(edge_vertices[:, :, 1]))

    w = maxx-minx
    h = maxy-miny
    padx,  pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    return edge_collection

def draw_nx_nodes(G, pos,
                  nodelist=None,
                  node_size=300,
                  node_color='r',
                  node_shape='o',
                  highlight=None,
                  cmap=None,
                  vmin=None,
                  vmax=None,
                  ax=None,
                  linewidths=0.5,
                  label=None,
                  **kwds):
    """Draw the nodes of the graph G.
    This draws only the nodes of the graph G.
    Parameters
    ----------
    G : graph
       A networkx graph
    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.
    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.
    nodelist : list, optional
       Draw only specified nodes (default G.nodes())
    node_size : scalar or array
       Size of nodes (default=300).  If an array is specified it must be the
       same length as nodelist.
    node_color : color string, or array of floats
       Node color. Can be a single color format string (default='r'),
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.
    node_shape :  string
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8' (default='o').
    highlight : array of boolean, if true a black ring is draw round the node
    cmap : Matplotlib colormap
       Colormap for mapping intensities of nodes (default=None)
    vmin,vmax : floats
       Minimum and maximum for node colormap scaling (default=None)
    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)
    label : [None| string]
       Label for legend
    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.
    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> nodes=nx.draw_networkx_nodes(G,pos=nx.spring_layout(G))
    Also see the NetworkX drawing examples at
    http://networkx.github.io/documentation/latest/gallery.html
    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)

    if not nodelist or len(nodelist) == 0:  # empty nodelist, no drawing
        return None

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise nx.NetworkXError('Node %s has no position.' % e)
    except ValueError:
        raise nx.NetworkXError('Bad value in node positions.')

    node_collection = ax.scatter(xy[:, 0], xy[:, 1],
                                 s=node_size,
                                 c=node_color,
                                 edgecolors='gray',
                                 marker=node_shape,
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax,
                                 alpha=1.0,
                                 linewidths=linewidths,
                                 label=label)
    node_collection.set_zorder(2)
    if highlight is not None and len(highlight) == len(xy):
        hxy = xy[highlight]
        hsz = node_size[highlight]
        highlight_collection = ax.scatter(hxy[:, 0], hxy[:, 1],
                                 s=20+2*hsz,
                                 c='none',
                                 edgecolors='k',
                                 marker='o')
        highlight_collection.set_zorder(2)

def getCategoricalColors(nw, df, node_attr, node_vals, color_edges, colors, edge_color='gray'):
    if colors is None:
        colors = ["red", "limegreen", "orange", "mediumblue", "yellow", "darkviolet",
                  "darkred", "darkgreen", "chocolate", "dodgerblue", "gold", "rebeccapurple",
                  "tomato", "seagreen", "navajowhite", "lightskyblue", "lemonchiffon", "plum"]
    counts = df[node_attr].value_counts()
    # color nodes by attribute
    if node_vals is not None:
        #attr_colors = {val: colors[idx%len(colors)] else 'gray' for idx, val in enumerate(node_vals)}
        attr_colors = {val: colors[idx] if idx < len(colors) else 'gray' for idx, val in enumerate(node_vals)}
    else:
        #attr_colors = {val: colors[idx%len(colors)] for idx, val in enumerate(counts.index)}
        attr_colors = {val: colors[idx] if idx < len(colors) else 'gray' for idx, val in enumerate(counts.index)}
    node_color_map = dict(zip(df['id'], df[node_attr].map(attr_colors)))
    node_colors = [node_color_map[node] for node in nw.nodes()]
    # color edges
    if color_edges:
        edge_colors = [node_color_map[edge[0]] if node_color_map[edge[0]] == node_color_map[edge[1]] else edge_color for edge in nw.edges()]
    else:
        edge_colors = [edge_color]*nw.number_of_edges()
    return np.array(node_colors), np.array(edge_colors), attr_colors

# draw network where nodes are colored by a categorical attribute node_attr
# nodes are ordered by descending frequency of category values
# optionally color_edges by whether they are intra (colored) or intra (gray) category
# optional list of node_attr values allows drawing network with predefined attribute value ordering
# so multiple images can have the same color scheme
# if plotfile is given, image is written to file
def draw_network_categorical(nw, df, node_attr='Cluster', node_vals=None, node_size=30,
                             plotfile=None, colors=None, title=None, color_edges=True, draw_edges=True):
    layout = dict(zip(df.id, list(zip(df.x, df.y))))
    node_colors, edge_colors, attr_colors = getCategoricalColors(nw, df, node_attr, node_vals, color_edges, None)
    # plot network
    fig = plt.figure(figsize=(10,8), tight_layout={'rect': (0, 0, 0.8, 1)})
    plt.axis('off')
    _draw_networkx_(nw, layout, arrow=False,
                     with_labels=False, node_color=node_colors,
                     edge_color=edge_colors, node_size=node_size, draw_edges=draw_edges)
    if title is not None:
        plt.gca().set_title(title)
    # add legend
    counts = df[node_attr].value_counts()
    patches = [mpatches.Patch(color=attr_colors[val], label=val+'('+str(counts.loc[val])+')') for val in counts.index]
    fig.axes[0].legend(handles=patches,loc='upper right', bbox_to_anchor=(1.25, 1.0), borderpad=0.05, borderaxespad=0.05)
    if plotfile is not None:
        plt.savefig(plotfile)

