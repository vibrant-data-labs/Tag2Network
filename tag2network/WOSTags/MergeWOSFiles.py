# -*- coding: utf-8 -*-

import os
import io
import glob

# concatenate a set of files Web of Science data files
# files are named filebas*.txt in the folder datapath
# output to outfile
#
def mergeWOSFiles(datapath, filebase, outfile):
    files = [f for f in glob.glob(os.path.join(datapath, filebase+"*.txt"))]

    of = io.open(os.path.join(datapath, outfile), 'w', encoding='utf-8')

    # concatenate files, ditch header from all files except the first one
    cnt = 0
    for idx, fname in enumerate(files):
        print("Processing file %d %s"%(idx, fname))
        df = io.open(fname, 'r', encoding='utf-8')
        for lnum, li in enumerate(df):
            if idx == 0 or lnum > 1:
                of.writelines([li[0:131000]])
                cnt += 1
        print("%d lines"%cnt)
        df.close()
    of.close()
    print("%d lines"%cnt)

