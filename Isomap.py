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
    
    #then do APSP
    aconfig = APSPConfig(dataSet,eps,GPU_MEM_SIZE)
    pathMatrix = APSP(knnList,aconfig)
    del knnList
    
    #then normalize the matrix
    normMatrix = NormMatrix(pathMatrix,GPU_MEM_SIZE)
    del pathMatrix
    
    #then get eigenvalues
    embedding = EigenEmbedding(normMatrix,trgDims)
    del normMatrix
    
    return embedding





if __name__ == '__main__':
    arg_values = ['nonmetric=','outdims=','indims=','if=','of=','k=','eps=','help','-h']
    
    optlist, args = getopt.getopt(sys.argv[1:], 'x', arg_values)
    
    trgDims = 3
    srcDims = 10000000000
    k = 8
    eps = 1000000000.
    infile='swissroll.csv'
    outfile='embedding.csv'
    nonmetric=False
    
    for o in optlist:
        if o[0].strip('-') == 'nonmetric':
            pass
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
    for o in optlist:
        if o[0].strip('-') == 'nonmetric':
            if o[1] == 'true':
                nonmetric = True
    for o in optlist:
        if o[0].strip('-') == 'help' or o[1].strip('-') == 'h':
            print "help not implemented."
    
    result = None
    if not nonmetric:
        result = Isomap(infile,srcDims,trgDims,k,eps)
    
    
    saveTable(result,outfile)
