# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 10:19:52 2014

@author: rich

add intercluster fraction, cluster diversity, bridging and centrality to each node

"""

import numpy as np


def basicClusteringProperties(network, clustering):
    """
    compute diversity and related properties for the given clustering
    adds results to node attributes
    """
    if clustering == 'Cluster':
        properties = ['InterclusterFraction', 'ClusterDiversity', 'ClusterBridging', 'ClusterArchetype']
    else:
        properties = ['fracIntergroup_' + clustering, 'diversity_' + clustering, 'bridging_' + clustering,
                      'centrality_' + clustering]
    results = {prop: {} for prop in properties}

    clusters = {}
    # iterate over each node
    for node in network:
        nodedata = network.node[node]
        cluster = nodedata[clustering] if clustering in nodedata else None
        if cluster != None and cluster != '':
            # build list of nodes in each cluster
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(node)
            clusterCounts = {}  # dict of cluster name and count
            nIntergroup = 0
            degree = float(network.degree(node))

            # walk neighbors and save cluster info
            for neighbor in network.neighbors_iter(node):
                neighborCluster = network.node[neighbor][clustering]
                if neighborCluster not in clusterCounts:
                    clusterCounts[neighborCluster] = 0.0
                clusterCounts[neighborCluster] += 1.0
                if neighborCluster is not None and cluster != neighborCluster:
                    nIntergroup += 1

            # compute diversity and related properties
            nGroups = len(clusterCounts)
            fracIntergroup = float(nIntergroup) / degree if (degree > 0) else 0
            diversity = 0
            p = np.array(clusterCounts.values()) / degree
            p = p[np.nonzero(p)]
            diversity = -np.sum(p * np.log(p))
            bridging = 0 if nGroups < 2 else diversity * float(nIntergroup) / (nGroups - 1)
            centrality = (1 - fracIntergroup) * degree / (1 + diversity)

            results[properties[0]][node] = fracIntergroup
            results[properties[1]][node] = diversity
            results[properties[2]][node] = bridging
            results[properties[3]][node] = centrality
    # normalize values within each cluster
    for nodes in clusters.itervalues():
        def normalize(results, nodes, attr):
            def normalizeArray(arr):
                npArr = np.array(arr)
                sd = np.std(npArr)
                if sd != 0:
                    mn = np.mean(npArr)
                    return ((npArr - mn) / sd).tolist()
                return arr

            vals = normalizeArray([results[attr][node] for node in nodes])
            for v in zip(nodes, vals):
                results[attr][v[0]] = v[1]

        normalize(results, nodes, properties[1])
        normalize(results, nodes, properties[2])
        normalize(results, nodes, properties[3])

    return results
