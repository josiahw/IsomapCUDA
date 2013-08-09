#! /usr/bin/env python
####################################################################################################################################################
#Copyright (c) 2013, Josiah Walker
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or #other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED #WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY #DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS #OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING #NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################################################################################################################################

"""
An interface and command line utility for running Isomap algorithms on datasets
"""


from IsomapCuda import *
from DataUtils import *
import getopt,sys

GPU_MEM_SIZE = 512

def Isomap(dataSet,srcDims,trgDims,k,eps=1000000000.):
    """
    Classical isomap
    """
    
    #first do KNN
    kconfig = KNNConfig(dataSet,k,eps,GPU_MEM_SIZE)
    kconfig['sourceDims'] = min(srcDims,kconfig['sourceDims'])
    knnList = KNN(dataSet,kconfig)
    #knnList = simpleKNN(dataSet,k,eps)
    
    
    
    #then do APSP
    aconfig = APSPConfig(knnList,eps,GPU_MEM_SIZE)
    pathMatrix = APSP(knnList,aconfig)
    del knnList
    
    #XXX:hacky way of saving this info
    saveTable(pathMatrix,'distances.csv')
    
    """
    #then normalize the matrix
    nconfig = NormMatrixConfig(pathMatrix,GPU_MEM_SIZE)
    normMatrix = NormMatrix(pathMatrix,nconfig)
    del pathMatrix
    
    #then get eigenvalues
    embedding = EigenEmbedding(normMatrix,trgDims)
    del normMatrix
    """
    
    #then get the rank matrix
    origDims = len(pathMatrix)
    rankMatrix = RankMatrix(pathMatrix)
    del pathMatrix
    
    #then get the NMDS embedding
    nconfig = NMDSConfig(rankMatrix,trgDims)
    nconfig['sourceDims'] = origDims
    embedding = CPUNMDS1(rankMatrix, loadMatrix(dataSet),nconfig)
    
    return embedding





if __name__ == '__main__':
    arg_values = ['nonmetric=','outdims=','indims=','if=','of=','k=','eps=','help','h']
    optlist, args = getopt.getopt(sys.argv[1:], 'x', arg_values)
    
    trgDims = 3
    srcDims = 10000000000
    k =6
    eps = 1000000000.
    infile='swissroll.csv'
    outfile='embedding.csv'
    nonmetric=False
    
    
    for o in optlist:
        if o[0].strip('-') == 'outdims':
            trgDims = int(o[1])
    for o in optlist:
        if o[0].strip('-') == 'indims':
            srcDims = int(o[1])
    for o in optlist:
        if o[0].strip('-') == 'if':
            infile = o[1]
    for o in optlist:
        if o[0].strip('-') == 'of':
            outfile = o[1]
    for o in optlist:
        if o[0].strip('-') == 'k':
            k = int(o[1])
    #for o in optlist:
    #    nonmetric = True
        
    for o in optlist:
        if o[0].strip('-') == 'help' or o[1].strip('-') == 'h':
            print "The following commands are available:"
            print "\t--if=inputfile\tDefaults to swissroll.csv"
            print "\t--of=outputfile\tDefaults to embedding.csv"
            print "\t--k=k_nearest_neighbours\tDefaults to 12"
            print "\t--outdims=embedding_dimensions\tDefaults to 3"
            print "\t--indims=input_dimensions\tDefaults to all in the input file"
            print "\t--nonmetric\tEnables non-metric MDS embeddings"
    result = None
    if not nonmetric:
        result = Isomap(infile,srcDims,trgDims,k,eps)
    
    
    saveTable(result,outfile)
    
