# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from collections import Counter
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix

import networkx as nx
import ClusteringProperties as cp
from DrawNetwork import draw_network_categorical
from louvain import generate_dendrogram
from louvain import partition_at_level
from tSNELayout import runTSNELayout

# build sparse feature matrix with optional idf weighting
# each row is a document, each column is a tag
# weighting assumes each term occurs once in each doc it appears in
def buildFeatures(df, tagHist, idf, tagAttr):
    allTags = tagHist.keys()
    # build tag-index mapping
    tagIdx = dict(zip(allTags, xrange(len(allTags))))
    # build feature matrix
    print("Build feature matrix")
    nDoc = len(df)
    features = dok_matrix((nDoc, len(tagIdx)), dtype=float)
    row = 0
    for tagList in df[tagAttr]:
        tagList = [k for k in tagList if k in tagIdx]
        if len(tagList) > 0:
            for tag in tagList:
                if idf:
                    docFreq = tagHist[tag]
                    features[row, tagIdx[tag]] = math.log(nDoc/float(docFreq), 2.0)
                else:
                    features[row, tagIdx[tag]] = 1.0
        else:
            print("Document with no tags (%s)"%tagAttr)
        row += 1
    return csr_matrix(features)

# compute cosine similarity
# f is a (sparse) feature matrix
def simCosine(f):
    # compute feature matrix dot product
    fdot = np.array(f.dot(f.T).todense())
    # get inverse feature vector magnitudes
    invMag = np.sqrt(np.array(1.0/np.diag(fdot)))
    # set NaN to zero
    invMag[np.isinf(invMag)] = 0
    # get cosine sim by elementwise multiply by inverse magnitudes
    sim = (fdot * invMag).T * invMag
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
def buildClusterNames(df, allTagHist, tagAttr, clAttr='Cluster', wtd=True):
    allVals = np.array(allTagHist.values(), dtype=float)
    allFreq = dict(zip(allTagHist.keys(), allVals/allVals.sum()))
    #clusters = df['clusId'].unique()
    clusters = df[clAttr].unique()
    df['cluster_name'] = ''
    clusInfo = []
    for clus in clusters:
        clusRows = df[clAttr] == clus
        nRows = clusRows.sum()
        if nRows > 0:
            tagHist = Counter([k for tagList in df[tagAttr][clusRows] for k in tagList if k in allTagHist])
            if wtd:
                vals = np.array(tagHist.values(), dtype=float)
                freq = dict(zip(tagHist.keys(), vals/vals.sum()))
                wtdTag = [(item[0], item[1]*math.sqrt(freq[item[0]]/allFreq[item[0]])) for item in tagHist.most_common()]
                wtdTag.sort(key=lambda x: x[1], reverse=True)
                topTag = [item[0] for item in wtdTag[:10]]
            else:
                topTag = [item[0] for item in tagHist.most_common()][:10]
            # remove unigrams that make up n-grams with n > 1
            topSet = set(topTag)
            removeTag = set()
            for tag in topTag:
                tags = tag.split(' ')
                if len(tags) > 1 and all(k in topSet for k in tags):
                    removeTag.update(tags)
            topTag = [k for k in topTag if k not in removeTag]
            # build and store name
            clName = ', '.join(topTag[:5])
            df.loc[clusRows,'cluster_name'] = clName
            clusInfo.append((clus, nRows, clName))
    clusInfo.sort(key=lambda x: x[1], reverse=True)
    for info in clusInfo:
        print("Cluster %s, %d nodes, name: %s"%info)

# build network, linking based on common tags, tag lists in column named tagAttr
def buildTagNetwork(df, tagAttr='eKwds', dropCols=[], outname=None,
                        nodesname=None, edgesname=None, plotname=None, idf=True,
                        toFile=True, doLayout=True, draw=False):
    print("Building document network")
    tagHist = dict([item for item in Counter([k for kwList in df[tagAttr] for k in kwList]).most_common() if item[1] > 1])
    # build document-keywords feature matrix
    features = buildFeatures(df, tagHist, idf, tagAttr)
    # compute similarity
    print("Compute similarity")
    sim = simCosine(features)
    # avoid self-links
    np.fill_diagonal(sim, 0)
    # threshold
    print("Threshold similarity")
    sim = threshold(sim)
    # make edge dataframe
    edgedf = simMatToLinkDataFrame(sim)
    df['id'] = range(len(df))
    df.drop(dropCols, axis=1, inplace=True)
    # add clusters and attributes
    nw = buildNetworkX(edgedf)
    addLouvainClusters(df, nw=nw)
    addNetworkAttributes(df, nw=nw)
    buildClusterNames(df, tagHist, tagAttr)
    if doLayout:
        layout = add_layout(df, nw=nw)
        if draw:
            draw_network_categorical(nw, df, layout, plotfilename=plotname)

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
def _add_attr(nodesdf, attr, vals):
    nodesdf[attr] = nodesdf['id'].map(vals)

# add network structural attributes to nodesdf
# clusVar is the attribute to use for computing bridging etc
def addNetworkAttributes(nodesdf, linksdf=None, nw=None, groupVars=["Cluster"], isDirected=False):
    if nw is None:
        nw = buildNetworkX(linksdf)
    if isDirected:
        _add_attr(nodesdf, "InDegree", dict(nw.in_degree()))
        _add_attr(nodesdf, "OutDegree", dict(nw.out_degree()))
    else:
        _add_attr(nodesdf, "Degree", dict(nw.degree()))

    # add bridging, cluster centrality etc. for one or more grouping variables
    for groupVar in groupVars:
        if len(nx.get_node_attributes(nw, groupVar)) == 0:
            vals = {k:v for k,v in dict(zip(nodesdf['id'], nodesdf[groupVar])).iteritems() if k in nw}
            nx.set_node_attributes(nw, vals, groupVar)
        grpprop = cp.basicClusteringProperties(nw, groupVar)
        for prop, vals in grpprop.iteritems():
            _add_attr(nodesdf, prop, vals)

# compute and add Louvain clusters to node dataframe
def addLouvainClusters(nodesdf, linksdf=None, nw=None, clusterLevel=0):
    def mergePartitionData(g, p, name):
        return {node: (name + '_' + str(p[node]) if node in p else None) for node in g.nodes()}

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
        _add_attr(nodesdf, grp, vals)

def add_layout(nodesdf, linksdf=None, nw=None):
    print("Running graph layout")
    if nw is None:
        nw = buildNetworkX(linksdf)
    layout, _ = runTSNELayout(nw)
    #layout = nx.spring_layout(nw, iterations=25)
    nodesdf['x'] = nodesdf['id'].apply(lambda x: layout[x][0] if x in layout else 0.0)
    nodesdf['y'] = nodesdf['id'].apply(lambda x: layout[x][1] if x in layout else 0.0)
    return layout

