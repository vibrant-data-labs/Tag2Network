# -*- coding: utf-8 -*-
#
#
# Build network from similarity matrix by thresholding
# or from matrix of possibly weighted connections 
#
# Add data to nodes - cluster, netowrk proerties, layout coordinates
#

import numpy as np
import pandas as pd
import math
from collections import Counter
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
import networkx as nx

import Network.ClusteringProperties as cp
from Network.DrawNetwork import draw_network_categorical
#from InteractiveNetworkViz import drawInteractiveNW
from Network.louvain import generate_dendrogram
from Network.louvain import partition_at_level
from Network.tSNELayout import runTSNELayout, runUMAPlayout
from Network.ClusterLayout import run_cluster_layout


# build sparse feature matrix with optional idf weighting
# each row is a document, each column is a tag
# weighting assumes each term occurs once in each doc it appears in
def buildFeatures(df, tagHist, idf, tagAttr):
    allTags = tagHist.keys()
    # build tag-index mapping
    tagIdx = dict(zip(allTags, range(len(allTags))))
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
            print("Document with no tags (%s)" % tagAttr)
        row += 1
    return csr_matrix(features)


# compute cosine similarity
# f is a (sparse) feature matrix
def simCosine(f):
    # compute feature matrix dot product
    fdot = np.array(f.dot(f.T).todense())
    # get inverse feature vector magnitudes
    invMag = np.sqrt(np.divide(1.0, np.diag(fdot)))
    # set NaN to zero
    invMag[np.isinf(invMag)] = 0
    # get cosine sim by elementwise multiply by inverse magnitudes
    sim = (fdot * invMag).T * invMag
    return sim


def threshold(sim, linksPer=4, connect_isolated_pairs=True):
    '''
    threshold similarity matrix - threshold by minimum maximum similarity of each node. Then if there 
    are too many links, thin links of each node propoertional to their abundance, but leave at least 
    one link from each node
    Parameters:
        sim - the similarity matrix
        linksPer - target connectivity
        connect_isolated_pairs - if true, connect isolated reciprocal pairs of nodes to their next 
        most similar neighbors
    '''
    simvals = sim.copy()
    nnodes = sim.shape[0]
    targetL = nnodes*linksPer
    # threshold on minimum max similarity in each row, this keeps at least one link per row
    mxsim = sim.max(axis=0)
    thr = mxsim[mxsim > 0].min()
    sim[sim < thr] = 0.0
    nL = (sim > 0).sum()
    # if too many links, keep equal fraction of links in each row,
    # minimally 1 per row, keep highest similarity links
    if nL > targetL:
        # get fraction to keep
        frac = targetL/float(nL)
        # sort the rows
        indices = np.argsort(sim, axis=1)
        # get number of elements in each row
        nonzero = np.round(np.maximum((frac*((sim > 0).sum(axis=1))), 1)).astype(int)
        # get minimum index to keep in each row
        minelement = (nnodes - nonzero).astype(int)
        # in each row, set values below number to keep to zero
        for i in range(nnodes):
            sim[i][indices[i][:minelement[i]]] = 0.0
        if connect_isolated_pairs:
            # for isolated reciprocal pairs, keep next lower similarity link
            # fisrt find all reciprocal pairs
            upper = np.triu(sim)
            recip = np.argwhere((upper > 0) & (np.isclose(upper, np.tril(sim).T, 1e-14)))
            # get isolated reciprocal pairs
            links = sim > 0
            isolated = (links[recip[:, 0]].sum(axis=1) == 1) & (links[recip[:, 1]].sum(axis=1) == 1) 
            # get all nodes involved in isolated pairs
            isolated_recip = recip[isolated].flatten()
            # add next most similar link        
            sim[isolated_recip, indices[isolated_recip, -2]] = simvals[isolated_recip, indices[isolated_recip, -2]]
    return sim


# build cluster name based on keywords that occur commonly in the cluster
# if wtd, then weigh keywords based on local frequency relative to global freq
# NOTE TO RICH:  I ADDED HERE PARAMETERS TO NAME THE CLUSTER NAME AND TOP TAGS COLUMNS
def buildClusterNames(df, allTagHist, tagAttr, 
                      clAttr='Cluster', clusterName='cluster_name', 
                      topTags='top_tags', wtd=True):
    allVals = np.array(list(allTagHist.values()), dtype=float)
    allFreq = dict(zip(allTagHist.keys(), allVals/allVals.sum()))
    #clusters = df['clusId'].unique()
    clusters = df[clAttr].unique()
    df[clusterName] = ''
    df[topTags] = ''
    clusInfo = []
    for clus in clusters:
        clusRows = df[clAttr] == clus
        nRows = clusRows.sum()
        if nRows > 0:
            tagHist = Counter([k for tagList in df[tagAttr][clusRows] for k in tagList if k in allTagHist])
            if wtd:
                vals = np.array(list(tagHist.values()), dtype=float)
                freq = dict(zip(tagHist.keys(), vals/vals.sum()))
                # weight tags, only include tag if it is more common than global
                wtdTag = [(item[0], item[1]*math.sqrt(freq[item[0]]/allFreq[item[0]]))
                          for item in tagHist.most_common() if freq[item[0]] > allFreq[item[0]]]
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
            df.loc[clusRows, clusterName] = clName
            df.loc[clusRows, topTags] = ', '.join(topTag)
            clusInfo.append((clus, nRows, clName))
    df[topTags] = df[topTags].str.split(',')
    clusInfo.sort(key=lambda x: x[1], reverse=True)
    for info in clusInfo:
        print("Cluster %s, %d nodes, name: %s" % info)


def buildNetworkFromNodesAndEdges(nodesdf, edgedf, outname=None,
                                  nodesname=None, edgesname=None, plotfile=None,
                                  doLayout=True, clusteredLayout=False,
                                  tagHist=None, tagAttr=None):
    # add clusters and attributes
    nw = buildNetworkX(edgedf)
    addLouvainClusters(nodesdf, nw=nw)
    addNetworkAttributes(nodesdf, nw=nw)
    if tagHist and tagAttr:
        buildClusterNames(nodesdf, tagHist, tagAttr)
    if doLayout:
        add_layout(nodesdf, nw=nw, clustered=clusteredLayout)
        if plotfile is not None:
            #drawInteractiveNW(df, nw=nw, plotfile=plotfile)
            draw_network_categorical(nw, nodesdf, plotfile=plotfile, draw_edges=False)
    # output to csv
    if nodesname is not None and edgesname is not None:
        print("Writing nodes and edges to files")
        nodesdf.to_csv(nodesname, index=False)
        edgedf.to_csv(edgesname, index=False)
    # output to xlsx
    if outname is not None:
        print("Writing network to file")
        writer = pd.ExcelWriter(outname)
        nodesdf.to_excel(writer, 'Nodes', index=False)
        edgedf.to_excel(writer, 'Links', index=False)
        writer.save()
    return nodesdf, edgedf


# build network helper function
# thresholds similarity, computes clusters and other attributes, names clusters if applicable
# draws network, saves plot and data to files
# return nodesdf, edgedf
def _buildNetworkHelper(df, sim, linksPer=4, outname=None,
                        nodesname=None, edgesname=None, plotfile=None,
                        doLayout=True, clusteredLayout=False, tagHist=None, tagAttr=None):
    # threshold
    if linksPer > 0:
        print("Threshold similarity")
        sim = threshold(sim, linksPer=linksPer)
    # make edge dataframe
    edgedf = matrixToLinkDataFrame(sim)

    return buildNetworkFromNodesAndEdges(df, edgedf, outname=outname,
                                         nodesname=nodesname, edgesname=edgesname, plotfile=plotfile,
                                         doLayout=doLayout, clusteredLayout=clusteredLayout,
                                         tagHist=tagHist, tagAttr=tagAttr)


# build network, linking based on common tags, tag lists in column named tagAttr
# color_attr - the attribute to color the nodes by
# outname - name of xlsx file to output network to
# nodesname - name of file for nodes csv
# edgesname - name of file for edge csv
# plotfile - name of file for plot image
# doLayout - if true, run layout
# draw - if True and if running layout, then draw the network and possibly save image to file (if plotfile is given)
# return nodesdf, edgedf
def buildTagNetwork(df, color_attr="Cluster", tagAttr='eKwds', dropCols=[], outname=None,
                    nodesname=None, edgesname=None, plotfile=None, idf=True,
                    toFile=True, doLayout=True, clusteredLayout=False, linksPer=4, minTags=0):
    print("Building document network")
    df = df.copy()  # so passed-in dataframe is not altered
    # make histogram of tag frequencies, only include tags with > 1 occurence
    tagHist = dict([item for item in Counter([k for kwList in df[tagAttr] 
                                              for k in kwList]).most_common() if item[1] > 1])
    # filter tags to only include 'active' tags - tags which occur twice or more in the doc set
    df[tagAttr] = df[tagAttr].apply(lambda x: [k for k in x if k in tagHist])
    # filter docs to only include docs with a minimum number of 'active' tags
    df = df[df[tagAttr].apply(lambda x: len(x) >= minTags)]
    # build document-keywords feature matrix
    features = buildFeatures(df, tagHist, idf, tagAttr)
    # compute similarity
    print("Compute similarity")
    sim = simCosine(features)
    # avoid self-links
    np.fill_diagonal(sim, 0)
    df['id'] = range(len(df))
    df.drop(dropCols, axis=1, inplace=True)
    return _buildNetworkHelper(df, sim, outname=outname,
                               nodesname=nodesname, edgesname=edgesname, plotfile=plotfile,
                               doLayout=doLayout, clusteredLayout=clusteredLayout,
                               tagHist=tagHist, tagAttr=tagAttr, linksPer=linksPer)


# build network given node dataframe and similarity matrix
# color_attr is the attribute to color the nodes by
# outname - name of xlsx file to output network to
# nodesname - name of file for nodes csv
# edgesname - name of file for edge csv
# plotfile - name of file for plot image
# doLayout - if true, run layout
# draw - if True and if running layout, then draw the network and possibly save image to file (if plotfile is given)
def buildSimilarityNetwork(df, sim, color_attr="Cluster", outname=None,
                           nodesname=None, edgesname=None, plotfile=None,
                           toFile=True, doLayout=True, linksPer=4):
    df['id'] = range(len(df))
    return _buildNetworkHelper(df, sim, outname=outname,
                               nodesname=nodesname, edgesname=edgesname, plotfile=plotfile,
                               toFile=toFile, doLayout=doLayout, linksPer=linksPer)


# build link dataframe from matrix where non-zero element is a link
def matrixToLinkDataFrame(mat, undirected=False, include_weights=True):
    if undirected:  # make symmetric then take upper triangle
        mat = np.triu(np.maximum(mat, mat.T))   
    links = np.transpose(np.nonzero(mat))
    linkList = [{'Source': l[0], 'Target': l[1]} for l in links]
    if include_weights:
        linkList = [{'Source': l[0], 'Target': l[1], 'weight': mat[l[0], l[1]]} for l in links]
    else:
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
            vals = {k: v for k, v in dict(zip(nodesdf['id'], nodesdf[groupVar])).items() if k in nw}
            nx.set_node_attributes(nw, vals, groupVar)
        grpprop = cp.basicClusteringProperties(nw, groupVar)
        for prop, vals in grpprop.items():
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
    for grp, vals in clusterings.items():
        _add_attr(nodesdf, grp, vals)
        nodesdf[grp].fillna('No Cluster', inplace=True)


def add_layout(nodesdf, linksdf=None, nw=None, clustered=True):
    print("Running graph layout")
    if nw is None:
        nw = buildNetworkX(linksdf)
    if clustered:
        # layout, _ = runTSNELayout(nw, nodesdf=nodesdf, cluster='Cluster')
        # layout, _ = runUMAPlayout(nw, nodesdf=nodesdf, cluster='Cluster')
        layout, _ = run_cluster_layout(nw, nodes_df=nodesdf, cluster='Cluster')
    else:
        layout, _ = runTSNELayout(nw)
        # layout, _ = runUMAPlayout(nw)
    nodesdf['x'] = nodesdf['id'].apply(lambda x: layout[x][0] if x in layout else 0.0)
    nodesdf['y'] = nodesdf['id'].apply(lambda x: layout[x][1] if x in layout else 0.0)
    return layout


def add_force_directed_layout(nodesdf, linksdf=None, nw=None, iterations=100):
    print("Running force-directed graph layout")
    if nw is None:
        nw = buildNetworkX(linksdf)
    layout = nx.spring_layout(nw, iterations=iterations)
    nodesdf['x_force_directed'] = nodesdf['id'].apply(lambda x: layout[x][0] if x in layout else 0.0)
    nodesdf['y_force_directed'] = nodesdf['id'].apply(lambda x: layout[x][1] if x in layout else 0.0)
    return layout
