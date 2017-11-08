# Tag2Network
Build similarity network from any dataset where a set of keywords or other tags is assigned to each document

The network is build by computing cosine simiarity between each document, based on their tag sets, and thresholding the similarities to produce a network

Also includes code, in WOSTags folder, to process Web of Science search results to extract keywords and prepare data to build a network

Example RunWOSExample.py builds a document network from Web of Science data
