# -*- coding: utf-8 -*-

import pandas as pd

#from DrawNetwork import draw_network_categorical
from Network.InteractiveNetworkViz import drawInteractiveNW

nodesfile = "tag2network/Data/Example/ExampleNodes.csv"
edgesfile = "tag2network/Data/Example/ExampleEdges.csv"

nodesdf = pd.read_csv(nodesfile)
edgesdf = pd.read_csv(edgesfile)

drawInteractiveNW(nodesdf, edgesdf=edgesdf, color_attr="Cluster", label_attr="name",
                  title="Interactive Network Visualization",
                  plotfile="Data/Example/TestNetwork", inline=False)
