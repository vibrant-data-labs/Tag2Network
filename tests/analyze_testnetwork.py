#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:02:15 2019

@author: ericlberlow
"""
#%%
import sys
sys.path.append("../@commonFunctions")
from Network.BuildNetwork import buildTagNetwork

import pandas as pd
pd.set_option('display.expand_frame_repr', False) # expand display of data columns if screen is wide
import os


datapath="data" 
namebase="test"
filename = os.path.join(datapath, namebase + "nodes.xlsx")

# set up output file names
nwname = os.path.join(datapath, namebase + "_network.xlsx")
plotname = os.path.join(datapath, namebase +"_plot.pdf")

#%%
df = pd.read_excel(filename)

# convert tags to list (list of violation ID's)
df['taglist'] = df['tags'].apply(lambda x: str(x).split("|"))
df['taglist2'] = df['tags2'].apply(lambda x: str(x).split("|"))
df['taglist3'] = df['tags3'].apply(lambda x: str(x).split("|"))
df['taglist4'] = df['tags4'].apply(lambda x: str(x).split("|"))


#%%  Build Facility Network
# columns to delete in the output
dropCols = []
# build network linked by violation similarity
nodesdf, edgedf = buildTagNetwork(df, tagAttr='taglist', dropCols=dropCols, outname=nwname,
                    nodesname=None, edgesname=None, plotfile=plotname,
                    doLayout=True, linksPer=20)









