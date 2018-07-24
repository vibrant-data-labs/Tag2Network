# -*- coding: utf-8 -*-

import math
import numpy as np
import networkx as nx
import scipy.sparse as sps
from sklearn.manifold import TSNE

# run tSNE to layout the nodes in 2D space
# dist is a distance matrix.  If None, distances are computed using shortest paths
# paths longer then maxdist are assumed to be "long" and set to 2*maxdist
# returns dict of {nodeid: [x,y]}
# offset increases minimum and so decreases relative distance between nodes, to hopefully spread tight clusters
def runTSNELayout(nw, dists=None, maxdist=5):
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
            dist[(dist == -1) & (cc > 0).toarray()] = math.log(1+idx)   # fill as yet unfilled path lengths
        dist[dist==-1] = 2.0*maxlen
        return dist

    perp = min(50, nw.number_of_nodes()/10)
    print("Running tSNE layout")
    # compute network shortest path lengths up to a maximum threshold - beyond that things are far away
    if dists is None:
        print("Computing shortest paths")
        dists = computeShortPathUpToMax(nx.adjacency_matrix(nw), maxdist)
    # compute tSNE
    print("Computing tSNE")
    layout = TSNE(n_components=2, metric='precomputed',
                  early_exaggeration=5, perplexity=perp).fit_transform(dists)
    # build the output data structure
    nodes = nw.nodes()
    nodeMap = dict(zip(nodes, range(len(nodes))))
    layout_dict = {node:layout[nodeMap[node]] for node in nodes}
    print("Done running tSNE layout")
    # return both networkx style dict and array of positions
    return layout_dict, layout
