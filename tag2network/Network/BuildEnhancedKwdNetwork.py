# -*- coding: utf-8 -*-
import os.path
from os.path import expanduser
import numpy as np
import pandas as pd
import igraph as ig
import math
import re
from collections import Counter
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


# for each document, split text into 1-, 2-, 3-grams using stopwords and punctuation to separate phrases
# add each ngram to keyword list if it is present in the master keyword list
def addTextKeywords(df, allKwds):
    stopwords = set(['a', 'the', 'this', 'that', 'and', 'or', 'of', 'not', 'is', 'in', 'it', 'its', 'but', 'what', 'with'])

    def findKeywords(ngrams):
        textkwds = set()
        for ngram in ngrams:
            if ngram in allKwds:
                textkwds.add(ngram)
            #else:
               #print("keyword not found")                
        return list(textkwds)
    
    def add23Grams(wordlist):
        #print ("making bigrams and trigrams from text ngrams")
        bigrams = [' '.join(wordlist[i:i+2]) for i in xrange(len(wordlist)-1)]
        trigrams = [' '.join(wordlist[i:i+3]) for i in xrange(len(wordlist)-2)]
        return bigrams+trigrams

    def getNGrams(text):      
        wordlists = []
        phrases = re.split('[,.;:!|?]+', text.lower())
        for phrase in phrases:
            words = re.split(' ', phrase)
            currentlist = []
            for w in words:
                if w in stopwords:
                    if len(currentlist) > 0:
                        wordlists.append(currentlist)
                        currentlist = []
                else:
                    currentlist.append(w)
            if len(currentlist) > 0:
                wordlists.append(currentlist)
        ngrams = []
        for wordlist in wordlists:
            #ngrams.extend(wordlist)  #include all unigrams
            ngrams.extend(add23Grams(wordlist))
        return ngrams

    print ("getting ngrams from text")
    textngrams = df['Text'].apply(lambda x: getNGrams(x))  #get 2-, 3-grams out of text
    print ("matching text ngrams to master keyword list to get text keywords")
    textkwds = textngrams.apply(lambda x: findKeywords(x)) #add as keywords if in master list
    newkwds = df['kwds'] + textkwds  #add ngrams from text if in master kwd list
    newkwds = newkwds.apply(lambda x: (set(x)))

    return list(newkwds)

# split multi-word keywords and for each group of fewer words, 
# reduce n-grams with n > 2 to a set of bigrams
# add ngramas a keyword if that keyword appears in the master keyword list

# original version
def enhanceKeywords(df, allKwds):
    def getSubKwd(newKwds, kwdStr, first):
        cnt = len(kwdStr)
        if cnt > 2:
            for i in xrange(cnt-2):
                getSubKwd(newKwds, kwdStr[i:cnt-1+i], False)
        if not first:
            kw = ' '.join(kwdStr)
            if kw in allKwds:
                newKwds.add(' '.join(kwdStr))
            
    enhancedKwds = []
    for kwds in df['nKwds']:
        newKwds = set()
        for kwd in kwds:
            newKwds.add(kwd)
            getSubKwd(newKwds, kwd.split(' '), True)
        enhancedKwds.append(list(newKwds))
        
    return enhancedKwds

 
       
# build sparse feature matrix with optional idf weighting
def buildFeatures(df, kwHist, idf):
    print("Build features")
    
    allKwds = kwHist.keys()
    # build kw-index mapping
    kwIdx = dict(zip(allKwds, xrange(len(allKwds))))
    # build feature matrix
    print("Build feature matrix")
    nDoc = len(df)
    features = dok_matrix((nDoc, len(kwIdx)), dtype=float)
    row = 0
    for kwList in df['kwds']:
        kwList = [k for k in kwList if k in kwIdx]
        if len(kwList)  > 0:
            for kwd in kwList:
                if idf:
                    docFreq = kwHist[kwd]
                    features[row, kwIdx[kwd]] = math.log(nDoc/float(docFreq))
                else:
                    features[row, kwIdx[kwd]] = 1.0
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

# if wtd, then weigh keywords based on local frequency relative to global freq
def buildClusterNames(df, allKwHist, wtd=True):
    allVals = np.array(allKwHist.values(), dtype=float)
    allFreq = dict(zip(allKwHist.keys(), allVals/allVals.sum()))
    clusters = df['clusId'].unique()
    df['Keyword_Cluster'] = ''
    clusInfo = []
    for clus in clusters:
        clusRows = df['clusId'] == clus
        nRows = clusRows.sum()
        if nRows > 0:
            kwHist = Counter([k for kwList in df['kwds'][clusRows] for k in kwList if k in allKwHist])
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
            # build and store name based on top n kwds
            clName = ', '.join(topKw[:5])
            df['Keyword_Cluster'][clusRows] = clName
            clusInfo.append((clus, nRows, clName))
    clusInfo.sort(key=lambda x: x[1], reverse=True)
    for info in clusInfo:
        print("Cluster %s, %d nodes, name: %s"%info)
        

# decorate, clean, and write excel file
def writeFile(df, nw, keepCols, outname):

    print("Clean and write output file")
    
    # clean kwds format from list to string with pipe separators
    df['cleaned_keywords'] = df['kwds'].apply(lambda x: '|'.join(x))
    df['new_keywords'] = df['nKwds'].apply(lambda x: '|'.join(x))
    df['enhanced_keywords'] = df['eKwds'].apply(lambda x: '|'.join(x))

    #remove kwd lists and text
    dropCols = ['Text', 'kwds', 'nKwds', 'eKwds']
    df.drop(dropCols, axis=1, inplace=True)   
    

    # re-order columns to keep
    df = df[keepCols]


    # output to xlsx
    print("Writing network to file")
    edgedf = pd.DataFrame(nw.get_edgelist(), columns=['Source', 'Target'])    
    writer = pd.ExcelWriter(outname)
    df.to_excel(writer,'Nodes',index=False)
    edgedf.to_excel(writer,'Links',index=False)
    writer.save()
    
    return df, edgedf


# build affinity netowrk based on keyword similarity, extract and name clusters, layout graph
def buildNetwork(df, keywords, idf=True, doLayout=True):
    nDoc = len(df)
    # compute histogram of keywords    
    kwHist = dict([item for item in Counter([k for kwList in keywords for k in kwList]).most_common() if item[1] > 1])
    print("%d Keywords that occur 2 or more times - for Features"%len(kwHist))

    #build features
    features = buildFeatures(df, kwHist, idf)
    
    print("Compute, and Threshold similarity")
    # compute similarity
    sim = simCosine(features)
    # threshold
    sim = threshold(sim)

 
    # build and partition network
    print("Build and partition network")
    nw = ig.Graph.Adjacency(sim.tolist(), mode="UNDIRECTED")
    nw.vs['name'] = range(sim.shape[0])
    partL = nw.community_multilevel(return_levels=False)

    # extract clusters and add cluster id and name to each node
    clL = partL.subgraphs()
    clL.sort(key=lambda x:x.vcount(), reverse=True)
    cls = dict([(v['name'],i) for i in range(len(clL)) if clL[i].vcount() > 1 for v in clL[i].vs])
    df['id'] = xrange(nDoc)
    df['clusId'] = df['id'].apply(lambda idx: str(cls[idx]) if idx in cls else None)
    buildClusterNames(df, kwHist)
    
    # layout the graph
    print("Running layout algo")
    df.set_index('id', drop=False)
    layout = nw.layout_fruchterman_reingold()
    coords = np.array(layout.coords)
    df['x'] = coords[:,0]
    df['y'] = coords[:,1]
   
    return nw, layout
 
        
def buildKeywords(df, blacklist):
    # get a list of all keywords 
    kwds = df['Keywords'].str.split('|')
    kwds = kwds.apply(lambda x: [s.strip() for s in x if len(s) > 2])
    kwds = kwds.apply(lambda x: [s for s in x if s not in blacklist]) 
    kwds = kwds.apply(lambda x: [s.lower() for s in x]) #make all lower case
    df['kwds'] = kwds # each row is a list of keywords
    
    # build keyword histogram for keywords (that occur twice or more)
    kwHist = dict([item for item in Counter([k for kwList in kwds for k in kwList]).most_common() if item[1] > 1])
    print("%d Original Keywords that occur 2 or more times"%len(kwHist))
    
    #add ngrams from title and abstract if they are in the master keyword list 
    print("adding new keywords to docs if they occur in the text")
    newkwds = addTextKeywords(df, set(kwHist.keys()))
    df['nKwds'] = newkwds
    nKwHist = dict([item for item in Counter([k for kwList in newkwds for k in kwList]).most_common() if item[1] > 1])
    print("%d New Keywords that occur 2 or more times"%len(nKwHist))


    #split multi-word keywords and add if in the master keyword list 
    print("enhancing multi-keywords with sub-keywords")
    enhancedKwds = enhanceKeywords(df, set(kwHist.keys()))
    df['eKwds'] = enhancedKwds
     
    # recompute histogram on enhanced set of keywords    
    kwHist = dict([item for item in Counter([k for kwList in enhancedKwds for k in kwList]).most_common() if item[1] > 1])
    print("%d Enhanced Keywords that occur 2 or more times"%len(kwHist))



# --------------------------------------------------------------
# End of Functions 
# --------------------------------------------------------------

# Define input and output file paths    
datapath = expanduser("~/Dropbox/@Python/DocumentNetworks/data")
fname = os.path.join(datapath, "Keystone_AbsTitle.txt")
outname = os.path.join(datapath, "Keystone_eKwd_Network.xlsx")


# Define blacklist keywords if applicable
blacklist = []

blacklist = set(['Environmental Sciences & Ecology', 
                 'Biodiversity & Conservation',  'Genetics &', 'Heredity',
                 'Life Sciences & Biomedicine', 'Zoology', 'Science & Technology - Other Topics', 
                 'keystone species', 'keystone', 'Life Sciences & Biomedicine - Other Topics',
                 'Environmental Sciences &', 'Ecology', 'based on', 'as keystone', 
                 'effect on', 'as keystone species', 'response to', 'an important', 
                 'to assess', 'relationship between', 'effects on', 'as an', 
                 'to identify', 'impact on', 'impacts on', 'implications for',
                 'influence on', 'relationships between', 'to an', 'depend on', 
                 'evidence for', 'exposed to', 'consequences for'
                 ])    


#define columns to keep for output file

keepCols = ['id', 'cleaned_keywords', 'new_keywords', 'enhanced_keywords', 'Authors', 'Year', 'Publication',  
             'TimesCited', 'CitesPerYr', 'log_Cites', 'log_CitesPerYr',
             'Title','Abstract', 'Keywords', 'SubjectCategory', 'DOI', 'PageCount', 
             'Keyword_Cluster', 'Label', 'x', 'y'
            ]

# read file and drop rows with no keywords
df = pd.read_csv(fname, sep='\t', header=0).dropna(subset=['Keywords'])


# ehnahce keywords from abstract and title texst
buildKeywords(df, blacklist) # enhance keywords

# build and layout network - specify which keyword list to use
nw, layout = buildNetwork(df, df['eKwds'], idf=False, doLayout=True)
#plot(nw, layout = layout)  ## TODO need to install cairo?

# decorate file
df['log_Cites'] = np.round((np.log(df['TimesCited']+1)),2)  # log transform and round to 2 decimals
df['CitesPerYr'] = df['TimesCited']/(2018-df['Year'])
df['log_CitesPerYr'] = np.round((np.log(df['CitesPerYr']+1)),2)
     
# write network to excel with nodes and links sheets
writeFile(df, nw, keepCols, outname) 

