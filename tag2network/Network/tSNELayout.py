# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import scipy.sparse as sps
from sklearn.manifold import TSNE

def runTSNELayout(nw, maxdist=5):
    # adj is adjacency matrix
    # compute shortest paths up to a max path length
    # fill all longer paths with twice the max
    def computeShortPathUpToMax(adj, maxlen):
        dist = np.empty(adj.shape)
        dist.fill(-1)   # initialize to -1 - any element with -1 hasn't been filled yet
        np.fill_diagonal(dist, 0)
        cc = sps.identity(dist.shape[0])
        for idx in range(1, maxlen+1):
            cc = cc.dot(adj)
            dist[(dist == -1) & (cc > 0).toarray()] = idx   # fill as yet unfilled path lengths
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
