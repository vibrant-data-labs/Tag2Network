# -*- coding: utf-8 -*-

import math
import numpy as np
import networkx as nx
import scipy.sparse as sps
from sklearn.manifold import TSNE
import umap


def _setup_layout_dists(nw, nodesdf, dists, maxdist, cluster):
    def computeShortPathUpToMax(adj, maxlen):
        dist = np.empty(adj.shape)
        dist.fill(-1)   # initialize to -1 - any element with -1 hasn't been filled yet
        np.fill_diagonal(dist, 0)
        cc = sps.identity(dist.shape[0])
        for idx in range(1, maxlen + 1):
            cc = cc.dot(adj)
            dist[(dist == -1) & (cc > 0).toarray()] = math.log(1 + idx)   # fill as yet unfilled path lengths
        dist[dist == -1] = 2.0 * maxlen
        return dist

    # compute network shortest path lengths up to a maximum threshold - beyond that things are far away
    if dists is None:
        print("Computing shortest paths")
        dists = computeShortPathUpToMax(nx.adjacency_matrix(nw), maxdist)
    # reduce path length if the nodes are in the same cluster
    clus = nodesdf.Cluster.copy().sort_index().to_numpy()
    if nodesdf is not None and cluster in nodesdf:
        adj = nx.adjacency_matrix(nw).todense()
        for edge in nw.edges:
            adj[edge] = clus[edge[0]] == clus[edge[1]]
        dists = dists - (adj / 1.5)
    return dists, clus


# run tSNE to layout the nodes in 2D space
# dist is a distance matrix.  If None, distances are computed using shortest paths
# paths longer then maxdist are assumed to be "long" and set to 2*maxdist
# returns dict of {nodeid: [x,y]}
# offset increases minimum and so decreases relative distance between nodes, to hopefully spread tight clusters
def runTSNELayout(nw, nodesdf=None, dists=None, maxdist=5, cluster=None):
    # adj is adjacency matrix
    # compute shortest paths up to a max path length
    # fill all longer paths with twice the max
    def initial_positions(all_clus):
        r = 10
        clus = np.unique(all_clus)
        n_clus = len(clus)
        d_phi = 2 * np.pi / n_clus
        clus_idx = dict(zip(clus, np.arange(n_clus)))
        phi = np.array([d_phi * clus_idx[cl] for cl in all_clus])
        pos = r * np.random.rand(len(all_clus))
        x_pos = pos * np.cos(phi)
        y_pos = pos * np.sin(phi)
        return np.stack([x_pos, y_pos]).T

    print("Running tSNE layout")
    dists, clus = _setup_layout_dists(nw, nodesdf, dists, maxdist, cluster)
    perp = min(50, nw.number_of_nodes()/10)
    # compute tSNE
    print("Computing tSNE")
    layout = TSNE(n_components=2, metric='precomputed', init=initial_positions(clus),
                  early_exaggeration=5, perplexity=perp).fit_transform(dists)
    # build the output data structure
    nodes = nw.nodes()
    nodeMap = dict(zip(nodes, range(len(nodes))))
    layout_dict = {node: layout[nodeMap[node]] for node in nodes}
    print("Done running tSNE layout")
    # return both networkx style dict and array of positions
    return layout_dict, layout


def runUMAPlayout(nw, nodesdf=None, dists=None, maxdist=5, cluster=None):
    print("Running UMAP layout")
    dists, clus = _setup_layout_dists(nw, nodesdf, dists, maxdist, cluster)
    model = umap.UMAP(metric='precomputed')
    layout = model.fit_transform(dists)
    # build the output data structure
    nodes = nw.nodes()
    nodeMap = dict(zip(nodes, range(len(nodes))))
    layout_dict = {node: layout[nodeMap[node]] for node in nodes}
    print("Done running UMAP layout")
    # return both networkx style dict and array of positions
    return layout_dict, layout
