#
# build similarity network from Web of Science data

import os

from tag2network.WOSTags.MergeWOSFiles import mergeWOSFiles
from tag2network.WOSTags.ProcessReferences import processRawWoSData
from tag2network.WOSTags.ProcessReferences import loadAndFilterDocuments
from tag2network.Tags.BuildKeywords import buildKeywords
from tag2network.Network.BuildNetwork import buildTagNetwork

# columns to delete in the output
dropCols = ['text', 'AuKeywords', 'KeywordsPlus', 'keywords']

namebase = "Example"         # dataset/project name
rawfilebase = "savedrecs"   # name base of raw, unjoined WoS data files
outfile = namebase+"Raw.txt"
basepath = "Data"
datapath = os.path.join(basepath, namebase)

# concatentate multiple WOS data files
mergeWOSFiles(datapath, rawfilebase, outfile)
# extract smaller set of useful data from raw WOS output
processRawWoSData(datapath, namebase)

# if desired, make keyword blacklist and whitelist
blacklist = set([])
whitelist = set([])

fname = os.path.join(datapath, namebase+"Final.txt")

# too many documents - keep a fraction of them by year and citation rate
# df = loadAndFilterDocuments(fname, 0.9)
df = loadAndFilterDocuments(fname, 0.1)    # for debugging so it runs faster

# set up output file names
nwname = os.path.join(datapath, namebase+".xlsx")
nodesname = os.path.join(datapath, namebase+"Nodes.csv")
edgesname = os.path.join(datapath, namebase+"Edges.csv")
plotname = os.path.join(datapath, namebase+"Plot.pdf")

# build and enhance the keywords, add to df
kwAttr = buildKeywords(df, blacklist, whitelist)


# build network linked by keyword similarity
buildTagNetwork(df, kwAttr, dropCols=dropCols, outname=nwname,
                nodesname=nodesname, edgesname=edgesname, plotfile=plotname,
                doLayout=True, clusteredLayout=True)  # , draw=True)
