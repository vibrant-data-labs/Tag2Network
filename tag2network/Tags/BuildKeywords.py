# -*- coding: utf-8 -*-

import re
from collections import Counter

# build keywords dataset with keywords and a block of text
# blacklist and whitelist are both sets
# kwAttr is name of keyword column
# keyword column is a string of | separated keywords
# for each document, split text into 1-, 2-, 3-grams using stopwords and punctuation to separate phrases
# add each ngram to keyword list if it is present in the master keyword list
# the master keyword list is all keywords from kwAttr plus the whitelist
# "enhance" # splits multi-word keywords and adds sub-keywords if in the master keyword list
# syndic is a synonym dictionary {synonym:commonTerm} pairs
# all_text == True keeps all text-derived ngrams instead of matching to master list
def buildKeywords(df, blacklist, whitelist, replace_with_space={}, kwAttr='keywords', txtAttr='text', 
                  syndic=None, addFromText=True, enhance=True, all_text=False, include_unigrams=False):
    def addTextKeywords(df, allKwds, all_text):
        stopwords = set(['a', 'an', 'the', 'this', 'that', 'and', 'or', 'of', 'not', 'at',
                         'is', 'in', 'it', 'its', 'but', 'what', 'with', 'as', 'to',
                         'why', 'are', 'do', 'from', 'for', 'on'])

        def findKeywords(ngrams):
            if all_text:
                return ngrams
            textkwds = set()
            for ngram in ngrams:
                if ngram in allKwds:
                    textkwds.add(ngram)
                #else:
                   #print("keyword not found")
            return list(textkwds)

        def add23Grams(wordlist):
            #print ("making bigrams and trigrams from text ngrams")
            bigrams = [' '.join(wordlist[i:i+2]) for i in range(len(wordlist)-1)]
            trigrams = [' '.join(wordlist[i:i+3]) for i in range(len(wordlist)-2)]
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
                if include_unigrams:
                    ngrams.extend(wordlist)  #include all unigrams
                ngrams.extend(add23Grams(wordlist))
            return ngrams

        print ("getting ngrams from text")
        textngrams = df[txtAttr].apply(lambda x: getNGrams(x))  #get 2-, 3-grams out of text
        print ("matching text ngrams to master keyword list to get text keywords")
        textkwds = textngrams.apply(lambda x: findKeywords(x)) #add as keywords if in master list
        newkwds = df.kwds + textkwds  #add ngrams from text if in master kwd list
        newkwds = newkwds.apply(lambda x: (set(x))) # make sure keywords are unique
        return list(newkwds)

    # for each document, split multi-word keywords and for each group of fewer words,
    # add it as a keyword if that keyword appears in the master keyword list
    # reduce n-grams with n > 2 to a set of bigrams
    def enhanceKeywords(df, allKwds):
        def getSubKwd(newKwds, kwdStr, first):
            cnt = len(kwdStr)
            if cnt > 2:
                for i in range(cnt-2):
                    getSubKwd(newKwds, kwdStr[i:cnt-1+i], False)
            if not first:
                kw = ' '.join(kwdStr)
                if kw in allKwds:
                    newKwds.add(' '.join(kwdStr))
        enhancedKwds = []
        for kwds in df['newKwds']:
            newKwds = set()
            for kwd in kwds:
                newKwds.add(kwd)
                getSubKwd(newKwds, kwd.split(), True)
            enhancedKwds.append(list(newKwds))
        return enhancedKwds

    # get a list of all keywords
    kwds = df[kwAttr].str.split('|')
    kwds = kwds.apply(lambda x: [s.strip() for s in x if len(s) > 2])   # only keep kwds with 3 or more characters
    kwds = kwds.apply(lambda x: [s for s in x if s not in blacklist])   # only keep keywords not in blacklist
    kwds = kwds.apply(lambda x: [s.lower() for s in x]) # make all lower case
    if len(replace_with_space) > 0:
        sub_re = '['+ ''.join(replace_with_space)  +']'
        kwds = kwds.apply(lambda x: [re.sub(sub_re, ' ', s) for s in x]) # add whitespace
    kwds = kwds.apply(lambda x: list(set(x))) # make sure keywords are unique
    df['kwds'] = kwds # each value is a list of keywords

    # build keyword histogram for keywords (that occur twice or more)
    kwHist = dict([item for item in Counter([k for kwList in kwds for k in kwList]).most_common() if item[1] > 1])
    print("%d Original Keywords that occur 2 or more times"%len(kwHist))

    if addFromText:
        #add ngrams from title and abstract if they are in the master keyword list
        masterKwds = set(kwHist.keys()).union(whitelist)
        #add search terms (synonyms) and their mapped common term to master list
        if syndic:
            masterKwds.update(syndic.keys(), syndic.values())
        print("adding new keywords to docs if they occur in the text (title+abstract)")
        newkwds = addTextKeywords(df, masterKwds, all_text)
        df['newKwds'] = newkwds
        nKwHist = dict([item for item in Counter([k for kwList in newkwds for k in kwList]).most_common() if item[1] > 1])
        print("%d New Keywords that occur 2 or more times"%len(nKwHist))
    else:
        df['newKwds'] = df['kwds']

    if enhance:
        # split multi-word keywords and add sub-keywords if in the master keyword list
        print("enhancing multi-keywords with sub-keywords")
        df['eKwds'] = enhanceKeywords(df, masterKwds)
    else:
        df['eKwds'] = df['newKwds']
    kwAttr = 'eKwds'
    
    if syndic:
        # synonym dictionary is {term, commonterm}; map terms to common term
        df[kwAttr] = df[kwAttr].apply(lambda x: list(set([syndic[kw] if kw in syndic else kw for kw in x])))

    # recompute histogram on enhanced set of keywords
    kwHist = dict([item for item in Counter([k for kwList in df[kwAttr] for k in kwList]).most_common() if item[1] > 1])
    print("%d Enhanced Keywords that occur 2 or more times"%len(kwHist))
    df['enhanced_keywords'] = df[kwAttr].apply(lambda x: '|'.join(x))
    df.drop(['newKwds', 'kwds'], axis=1, inplace=True)
    return kwAttr
