# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE

def runTSNELayout(nw, maxdist=5):
    # adj is adjacency matrix
    # compute shortest paths up to a max path length
    # fill all longer paths with twice the max
    def computeShortPathUpToMax(adj, maxlen):
        dist = np.empty(adj.shape)
        dist.fill(-1)
        np.fill_diagonal(dist, 0)
        cc = adj
        for idx in range(1, maxlen):
            dist[(dist == -1) & (cc > 0).toarray()] = idx
            cc = cc.dot(adj)
        dist[dist==-1] = 2.0*maxlen
        return dist

    print("Running tSNE layout")

    # compute network shortest path lengths up to a maximum threshold - beyond that things are far away
    print("Computing shortest paths")
    dists = computeShortPathUpToMax(nx.adjacency_matrix(nw), maxdist)
    # compute tSNE
    print("Computing tSNE")
    layout = TSNE(n_components=2, metric='precomputed').fit_transform(dists)
    # build the output data structure
    nodes = nw.nodes()
    nodeMap = dict(zip(nodes, range(len(nodes))))
    layout_dict = {node:layout[nodeMap[node]] for node in nodes}
    print("Done running tSNE layout")
    return layout_dict
