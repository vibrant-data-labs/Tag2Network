# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from collections import Counter
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix

import networkx as nx
import ClusteringProperties as cp
from louvain import generate_dendrogram
from louvain import partition_at_level

# build sparse feature matrix with optional idf weighting
# each row is a document, each column is a keyword
# weighting assumes each term occurs once in each doc it appears in
def buildFeatures(df, kwHist, idf, kwAttr):
    allKwds = kwHist.keys()
    # build kw-index mapping
    kwIdx = dict(zip(allKwds, xrange(len(allKwds))))
    # build feature matrix
    print("Build feature matrix")
    nDoc = len(df)
    features = dok_matrix((nDoc, len(kwIdx)), dtype=float)
    row = 0
    for kwList in df[kwAttr]:
        kwList = [k for k in kwList if k in kwIdx]
        if len(kwList) > 0:
            for kwd in kwList:
                if idf:
                    docFreq = kwHist[kwd]
                    features[row, kwIdx[kwd]] = math.log(nDoc/float(docFreq), 2.0)
                else:
                    features[row, kwIdx[kwd]] = 1.0
        else:
            print("Document with no extended keywords")
        row += 1
    return csr_matrix(features)

# compute cosine similarity
# f is a (sparse) feature matrix
def simCosine(f):
    fdot = np.array(f.dot(f.T).todense())
    # get inverse feature vector magnitudes
    invMag = np.sqrt(np.array(1.0/np.diag(fdot)))
    # set NaN to zero
    invMag[np.isinf(invMag)] = 0
    # get cosine sim by elementwise multiply by inverse magnitudes
    sim = (fdot * invMag).T * invMag
    np.fill_diagonal(sim, 0)
    return sim

def threshold(sim, linksPer=4):
    nnodes = sim.shape[0]
    targetL = nnodes*linksPer
    # threshold on minimum max similarity in each row, this keeps at least one link per row
    mxsim = sim.max(axis=0)
    thr = mxsim[mxsim>0].min()
    sim[sim < thr] = 0.0
    sim[sim >= thr] = 1.0
    nL = sim.sum()
    # if too many links, keep equal fraction of links in each row,
    # minimally 1 per row, keep highest similarity links
    if nL > targetL:
        # get fraction to keep
        frac = targetL/float(nL)
        # sort the rows
        indices = np.argsort(sim, axis=1)
        # get number of elements in each row
        nonzero = np.maximum(np.rint(frac * np.apply_along_axis(np.count_nonzero, 1, sim)), 1)
        # get minimum index to keep in each row
        minelement = (nnodes - nonzero).astype(int)
        # in each row, set values below number to keep to zero
        for i in xrange(nnodes):
            sim[i][indices[i][:minelement[i]]] = 0.0
    return sim

# build cluster name based on keywords that occur commonly in the cluster
# if wtd, then weigh keywords based on local frequency relative to global freq
def buildClusterNames(df, allKwHist, kwAttr, clAttr='Cluster', wtd=True):
    allVals = np.array(allKwHist.values(), dtype=float)
    allFreq = dict(zip(allKwHist.keys(), allVals/allVals.sum()))
    #clusters = df['clusId'].unique()
    clusters = df[clAttr].unique()
    df['cluster_name'] = ''
    clusInfo = []
    for clus in clusters:
        clusRows = df[clAttr] == clus
        nRows = clusRows.sum()
        if nRows > 0:
            kwHist = Counter([k for kwList in df[kwAttr][clusRows] for k in kwList if k in allKwHist])
            if wtd:
                vals = np.array(kwHist.values(), dtype=float)
                freq = dict(zip(kwHist.keys(), vals/vals.sum()))
                wtdKw = [(item[0], item[1]*math.sqrt(freq[item[0]]/allFreq[item[0]])) for item in kwHist.most_common()]
                wtdKw.sort(key=lambda x: x[1], reverse=True)
                topKw = [item[0] for item in wtdKw[:10]]
            else:
                topKw = [item[0] for item in kwHist.most_common()][:10]
            # remove unigrams that make up n-grams with n > 1
            topSet = set(topKw)
            removeKw = set()
            for kw in topKw:
                kws = kw.split(' ')
                if len(kws) > 1 and all(k in topSet for k in kws):
                    removeKw.update(kws)
            topKw = [k for k in topKw if k not in removeKw]
            # build and store name
            clName = ', '.join(topKw[:5])
            df.cluster_name[clusRows] = clName
            clusInfo.append((clus, nRows, clName))
    clusInfo.sort(key=lambda x: x[1], reverse=True)
    for info in clusInfo:
        print("Cluster %s, %d nodes, name: %s"%info)

# build network, linking based on common keywords, keyword lists in column named kwAttr
def buildKeywordNetwork(df, kwAttr='eKwds', dropCols=[], outname=None, nodesname=None, edgesname=None, idf=True, toFile=True, doLayout=True):
    print("Building document network")
    kwHist = dict([item for item in Counter([k for kwList in df[kwAttr] for k in kwList]).most_common() if item[1] > 1])
    # build document-keywords feature matrix
    features = buildFeatures(df, kwHist, idf, kwAttr)
    # compute similarity
    print("Compute similarity")
    sim = simCosine(features)
    # threshold
    print("Threshold similarity")
    sim = threshold(sim)
    # make edge dataframe and clean up node dataframe
    edgedf = simMatToLinkDataFrame(sim)
    df['id'] = range(len(df))
    df.drop(dropCols, axis=1, inplace=True)
    # add clusters and attributes
    nw = buildNetworkX(edgedf)
    addLouvainClusters(df, nw=nw)
    addNetworkAttributes(df, nw=nw)
    buildClusterNames(df, kwHist, kwAttr)
    if doLayout:
        addLayout(df, nw=nw)

    if toFile:
        # output to xlsx
        if outname is not None:
            print("Writing network to file")
            writer = pd.ExcelWriter(outname)
            df.to_excel(writer,'Nodes',index=False)
            edgedf.to_excel(writer,'Links',index=False)
            writer.save()
        # output to csv
        if nodesname is not None and edgesname is not None:
            print("Writing nodes and edges to files")
            df.to_csv(nodesname,index=False)
            edgedf.to_csv(edgesname,index=False)

# build link dataframe
def simMatToLinkDataFrame(simMat):
    links = np.transpose(np.nonzero(simMat))
    linkList = [{'Source': l[0], 'Target': l[1]} for l in links]
    return pd.DataFrame(linkList)

def buildNetworkX(linksdf, id1='Source', id2='Target', directed=False):
    linkdata = [(getattr(link, id1), getattr(link, id2)) for link in linksdf.itertuples()]
    g = nx.DiGraph() if directed else nx.Graph()
    g.add_edges_from(linkdata)
    return g

# add a computed network attribute to the node attribute table
#
def addAttr(nodesdf, attr, vals):
    nodesdf[attr] = nodesdf['id'].map(vals)

# add network structural attributes to nodesdf
# clusVar is the attribute to use for computing bridging etc
def addNetworkAttributes(nodesdf, linksdf=None, nw=None, groupVars=["Cluster"], isDirected=False):
    if nw is None:
        nw = buildNetworkX(linksdf)
    if isDirected:
        addAttr(nodesdf, "InDegree", nw.in_degree())
        addAttr(nodesdf, "OutDegree", nw.out_degree())
    else:
        addAttr(nodesdf, "Degree", nw.degree())

    # add bridging, cluster centrality etc. for one or more grouping variables
    for groupVar in groupVars:
        if len(nx.get_node_attributes(nw, groupVar)) == 0:
            vals = {k:v for k,v in dict(zip(nodesdf['id'], nodesdf[groupVar])).iteritems() if k in nw}
            nx.set_node_attributes(nw, groupVar, vals)
        grpprop = cp.basicClusteringProperties(nw, groupVar)
        for prop, vals in grpprop.iteritems():
            addAttr(nodesdf, prop, vals)

# compute and add Louvain clusters to node dataframe
def addLouvainClusters(nodesdf, linksdf=None, nw=None, clusterLevel=0):
    def mergePartitionData(g, p, name):
        return {node: (name + '_' + str(p[node]) if node in p else None) for node in g.nodes_iter()}

    def getPartitioning(i, g, dendo, clusterings):
        p = partition_at_level(dendo, len(dendo) - 1 - i)
        clustering = "Cluster"
        vals = mergePartitionData(g, p, clustering)
        clusterings[clustering] = vals

    if nw is None:
        nw = buildNetworkX(linksdf)
    print("Computing clustering")
    if isinstance(nw, nx.DiGraph):
        gg = nx.Graph(nw)
    else:
        gg = nw
    clusterings = {}
    dendo = generate_dendrogram(gg)
    #    for i in range(len(dendo)):
    #        getPartitioning(i, g, dendo, clusterings)
    depth = min(clusterLevel, len(dendo) - 1)
    getPartitioning(depth, gg, dendo, clusterings)
    # add cluster attr to dataframe
    for grp, vals in clusterings.iteritems():
        addAttr(nodesdf, grp, vals)

def addLayout(nodesdf, linksdf=None, nw=None):
    print("Running graph layout")
    if nw is None:
        nw = buildNetworkX(linksdf)
    layout = nx.spring_layout(nw, iterations=25)
    nodesdf['x'] = nodesdf['id'].apply(lambda x: layout[x][0] if x in layout else 0.0)
    nodesdf['y'] = nodesdf['id'].apply(lambda x: layout[x][1] if x in layout else 0.0)