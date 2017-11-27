# -*- coding: utf-8 -*-

import re
from collections import Counter

# build keywords from Web of Science data
# blacklist and whitelist are both sets
#
# for each document, split text into 1-, 2-, 3-grams using stopwords and punctuation to separate phrases
# add each ngram to keyword list if it is present in the master keyword list allKwds
# syndic is a synonym dictionary {synonym:commonTerm} pairs
def buildKeywords(df, blacklist, whitelist, kwAttr='keywords', txtAttr='text', syndic=None, addFromText=True, enhance=True):
    def addTextKeywords(df, allKwds):
        stopwords = set(['a', 'the', 'this', 'that', 'and', 'or', 'of', 'not',
                         'is', 'in', 'it', 'its', 'but', 'what', 'with'])

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
        textngrams = df[txtAttr].apply(lambda x: getNGrams(x))  #get 2-, 3-grams out of text
        print ("matching text ngrams to master keyword list to get text keywords")
        textkwds = textngrams.apply(lambda x: findKeywords(x)) #add as keywords if in master list
        newkwds = df.kwds + textkwds  #add ngrams from text if in master kwd list
        newkwds = newkwds.apply(lambda x: (set(x)))
        return list(newkwds)

    # for each document, split multi-word keywords and for each group of fewer words,
    # add it as a keyword if that keyword appears in the master keyword list
    # reduce n-grams with n > 2 to a set of bigrams
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

    # get a list of all keywords
    kwds = df[kwAttr].str.split('|')
    kwds = kwds.apply(lambda x: [s.strip() for s in x if len(s) > 2])   # only keep kwds with 3 or more characters
    kwds = kwds.apply(lambda x: [s for s in x if s not in blacklist])   # only keep keywords not in blacklist
    kwds = kwds.apply(lambda x: [s.lower() for s in x]) # make all lower case
    kwds = kwds.apply(lambda x: list(set(x))) # make sure keywords are unique
    df['kwds'] = kwds # each row is a list of keywords

    # build keyword histogram for keywords (that occur twice or more)
    kwHist = dict([item for item in Counter([k for kwList in kwds for k in kwList]).most_common() if item[1] > 1])
    print("%d Original Keywords that occur 2 or more times"%len(kwHist))

    if addFromText:
        #add ngrams from title and abstract if they are in the master keyword list
        masterKwds = set(kwHist.keys()).union(whitelist)
        print("adding new keywords to docs if they occur in the text (title+abstract)")
        newkwds = addTextKeywords(df, masterKwds)
        df['nKwds'] = newkwds
        nKwHist = dict([item for item in Counter([k for kwList in newkwds for k in kwList]).most_common() if item[1] > 1])
        print("%d New Keywords that occur 2 or more times"%len(nKwHist))
    else:
        df['nKwds'] = df['kwds']

    kwAttr = 'eKwds'
    if enhance:
        #split multi-word keywords and add if in the master keyword list
        print("enhancing multi-keywords with sub-keywords")
        df[kwAttr] = enhanceKeywords(df, masterKwds)
    else:
        df[kwAttr] = df['nKwds']

    # recompute histogram on enhanced set of keywords
    kwHist = dict([item for item in Counter([k for kwList in df[kwAttr] for k in kwList]).most_common() if item[1] > 1])
    print("%d Enhanced Keywords that occur 2 or more times"%len(kwHist))
    df['enhanced_keywords'] = df['eKwds'].apply(lambda x: '|'.join(x))
    df.drop(['nKwds', 'kwds'], axis=1, inplace=True)
    return kwAttr
