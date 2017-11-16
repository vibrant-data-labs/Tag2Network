# -*- coding: utf-8 -*-
#
# takes a Web Of Science tab delimited output file and turns it into a tab delimited file ready for text2network
#
# output text: abs - abstract only; all - text+title+keywords; wtd 3x title, 3x keywords 1x text; big - 3x all
#
"""
Created on Wed Mar 26 15:27:02 2014

@author: rich
"""

import os
import csv
import math
import pandas as pd

# extract useful columns, merge text fields, give columns readable names
def processRawWoSData(path, namebase):
    #pubTypes = ['conference', 'book', 'journal', 'book in series', 'patent']

    fieldInfo = {
    'FN':	{'name': 'FileName', 'use': False},
    'VR':	{'name': 'VersionNumber', 'use': False},
    'PT':	{'name': 'PubType', 'use': False},
    'AU':	{'name': 'Authors', 'use': True},
    'AF':	{'name': 'AuthorFullName', 'use': False},
    'CA':	{'name': 'GroupAuthors', 'use': False},
    'TI':	{'name': 'name', 'use': True},
    'ED':	{'name': 'Editors', 'use': True},
    'SO':	{'name': 'PubName', 'use': True},
    'SE':	{'name': 'Book Series Title', 'use': False},
    'BS':	{'name': 'Book Series Subtitle', 'use': False},
    'LA':	{'name': 'Language', 'use': False},
    'DT':	{'name': 'Document Type', 'use': False},
    'CT':	{'name': 'ConferenceTitle', 'use': False},
    'CY':	{'name': 'ConferenceDate', 'use': False},
    'HO':	{'name': 'ConferenceHost', 'use': False},
    'CL':	{'name': 'ConferenceLocation', 'use': False},
    'SP':	{'name': 'ConferenceSponsors', 'use': False},
    'DE':	{'name': 'AuKeywords', 'use': True},
    'ID':	{'name': 'KeywordsPlus', 'use': True},
    'AB':	{'name': 'Abstract', 'use': True},
    'C1':	{'name': 'AuthorAddress', 'use': False},
    'RP':	{'name': 'ReprintAddress', 'use': False},
    'EM':	{'name': 'E-mailAddress', 'use': False},
    'FU':	{'name': 'FunderAndGrant', 'use': False},
    'FX':	{'name': 'FundingText', 'use': False},
    'CR':	{'name': 'Cited References', 'use': False},
    'NR':	{'name': 'ReferenceCount', 'use': True},
    'TC':	{'name': 'TimesCited', 'use': True},
    'PU':	{'name': 'Publisher', 'use': False},
    'PI':	{'name': 'Publisher City', 'use': False},
    'PA':	{'name': 'Publisher Address', 'use': False},
    'SN':	{'name': 'ISSN', 'use': False},
    'BN':	{'name': 'ISBN', 'use': False},
    'J9':	{'name': 'SourceAbbrev', 'use': False},
    'JI':	{'name': 'ISOSourceAbbrev', 'use': False},
    'PD':	{'name': 'PubDate', 'use': False},
    'PY':	{'name': 'PubYear', 'use': True},
    'VL':	{'name': 'Volume', 'use': False},
    'IS':	{'name': 'Issue', 'use': False},
    'PN':	{'name': 'PartNumber', 'use': False},
    'SU':	{'name': 'Supplement', 'use': False},
    'SI':	{'name': 'SpecialIssue', 'use': False},
    'BP':	{'name': 'BeginningPage', 'use': False},
    'EP':	{'name': 'EndingPage', 'use': False},
    'AR':	{'name': 'ArticleNumber', 'use': False},
    'PG':	{'name': 'PageCount', 'use': True},
    'DI':	{'name': 'DOI', 'use': True},
    'SC':	{'name': 'SubjectCategory', 'use': True},
    'GA':	{'name': 'DocDeliveryNumber', 'use': False},
    'UT':	{'name': 'UniqueArticleIdentifier', 'use': False},
    'PM':	{'name': 'PubMedID', 'use': False},
    'ER':	{'name': 'EndRecord', 'use': False},
    'EF':	{'name': 'EndFile', 'use': False},
    }

    title = 'TI'
    kwds = 'DE'
    kwds2 = 'ID'
    pubDate = 'PD'
    pubYr = 'PY'
    abstract = 'AB'
    refs = 'CR'
    doi = 'DI'
    sType = 'SearchType'
    sRef = 'SReferences'
    xRef = 'XReferences'
    sxRef = 'SXReferences'
    inbred = 'Inbredness'

    #os.chdir(path)
    dataFile = os.path.join(path, namebase + "Raw.txt")

    # compute reference similarity for each pair of nodes
    def computeReferenceSimilarities(references):
        similarities = {}
        maxSim = 0
        for nodeid, ref in references.iteritems():
            sim = similarities[nodeid] = {}
            refSet = set(ref.split(';'))
            for nodeid2, ref2 in references.iteritems():
                if nodeid < nodeid2:
                    refSet2 = set(ref2.split(';'))
                    nTotal = len(refSet.union(refSet2))
                    nOverlap = len(refSet.intersection(refSet2))
                    if nOverlap > 0 and nTotal > 0:
                        simVal = float(nOverlap)/nTotal
                        sim[nodeid2] = simVal
                        if simVal > maxSim:
                            maxSim = simVal
        return similarities

    # write data, pass in filename and field to use for 'text' in final output
    def writeData(name, fields, data, textField):
        for row in data:
            row['text'] = row[textField]
        f = open(name, 'w')
        writer = csv.DictWriter(f, fields, dialect=csv.excel_tab, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)
        f.close()

    # main entry point for processing results from web of science searches.
    # hasSearchType is true if WoS results have an added searchType column,
    # used to combine and differentiate results from multiple searches
    def processWoSData(dataFile, hasSearchType):
        # get used field names
        fields = []
        for key, value in fieldInfo.iteritems():
            if value['use'] is True:
                fields.append(key)

        # read in and re-arrange the data
        data = []
        doiIndex = {}
        references = {}
        f = open(dataFile, 'rU')
        reader = csv.DictReader(f, dialect=csv.excel_tab)
        #read data one row at a time
        nodeid = 0
        for row in reader:
            # only include records that have abstract text and keywords
            if len(row[abstract]) > 0:
                if nodeid%100 == 0:
                    print("Processed %d"%nodeid)
                kwplus = row[kwds2]
                if len(kwplus) > 0 :
                    kwplus = kwplus.lower()
                dataRow = {}
                # build keywords
                keywords = row[kwds]
                if len(keywords) > 0 and len(kwplus) > 0:
                    keywords += ';'
                keywords += kwplus
                if len(keywords) == 0:
                    continue
                keywords = keywords.lower().replace(';', '|')
                # build text and add to row
                dataRow['alltext'] = row[title] + '. ' + row[abstract] + '. ' + keywords + '.'
                #dataRow['wtdtext'] = row[title] + '. ' + row[title] + '. ' + row[title] + '. ' + row[abstract] + '. ' + keywords + '.' + keywords + '.' + keywords + '.'
                #dataRow['bigtext'] = dataRow['alltext'] + ' ' + dataRow['alltext'] + ' ' + dataRow['alltext']
                # build date string and add to row
                date = ''
                if len(row[pubDate]) > 0:
                    date = row[pubDate] + '-'
                date += row[pubYr]
                dataRow['date'] = date
                dataRow['id'] = nodeid
                dataRow['keywords'] = keywords
                if hasSearchType:
                    dataRow[sType] = row[sType]
                doiIndex[row[doi]] = nodeid
                references[nodeid] = row[refs]
                nodeid += 1
                # copy 'used' fields into row with renamed output fields instead of WoS codes
                for field in fields:
                    info = fieldInfo[field]
                    if field in row:
                        dataRow[info['name']] = row[field]
                data.append(dataRow)
        f.close()

        if hasSearchType:
            # count references to papers with different searchTypes
            for nodeid, ref in references.iteritems():
                nS = 0
                nX = 0
                nSX = 0
                for refItem in ref.split(';'):
                    refFields = refItem.split(',')
                    doiField = refFields[len(refFields) - 1].lstrip()
                    if doiField.startswith("DOI"):
                        refDOI = doiField[4:]   # got a DOI reference
                        if doiIndex.has_key(refDOI):    # reference is in this data set
                            id = doiIndex[refDOI]
                            searchType = data[id][sType]
                            if searchType is 'S':
                                nS += 1
                            elif searchType is 'X':
                                nX += 1
                            else:
                                nSX += 1
                dataRow = data[nodeid]
                dataRow[sRef] = nS
                dataRow[xRef] = nX
                dataRow[sxRef] = nSX
                searchType = dataRow[sType]
                tot = nS+nX
                if tot > 3:     # no significant inbreeding when total number of s or x references is small
                    inbredVal = (2 * nS/tot) - 1        # -1 - all x, +1 - all s
                else:
                    inbredVal = 0
                dataRow[inbred] = inbredVal

        # build array of output fields
        outFields = []
        outFields.append('text')
        outFields.append('date')
        outFields.append('keywords')
        for field in fields:
            outFields.append(fieldInfo[field]['name'])
        if hasSearchType:
            outFields.append(sType)
            outFields.append(sRef)
            outFields.append(xRef)
            outFields.append(sxRef)
            outFields.append(inbred)
        # write out the data
        writeData(os.path.join(path, namebase + "Final.txt"), outFields, data, 'alltext')

    processWoSData(dataFile, False)

# keep a fraction of documents by year and citation rate
# sort by publication year and citation count
# then iterate through each year keeping documents in each year with most citations
def filterDocuments(fname, keepFrac):
    df = pd.read_csv(fname, sep='\t', header=0)

    if keepFrac < 1.0:
        # sort by year and citation count
        df.sort_values(by=['PubYear', 'TimesCited'], inplace=True, ascending=False)
        cnts = df.PubYear.value_counts()
        start = 0
        keep = pd.Series([False]*len(df))
        while start < len(df):
            cnt = cnts[df.PubYear[start]]
            last = int(start + cnt*math.ceil(keepFrac))
            keep[start:last] = True
            start += cnt
        return df[keep.values]
    return df

