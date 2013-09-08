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
GPU based K nearest neighbours algorithm.
"""

import time
from numpy import array,zeros,amax,amin,sqrt,dot,random
import numpy
from numpy.linalg import eig
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math


from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,loadIntMatrix,saveTable

KernelLocation = "CudaKernels/KNN/"

# KNN Algorithm --------------------------------------------

def GPUConfig(settings,memPerElement,memPerElementSq,limit=1000000000):
    
    
    return settings

def KNNConfig(dataTable,srcDims, k, eps = 1000000000.,gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated KNN.
    """
    kmax = 2**int(math.ceil(math.log(k+1)/math.log(2)))+1
    settings["oldk"] = k
    settings["kmax"] = kmax
    k = kmax
    settings = dataConfig(dataTable,settings)
    settings["sourceDims"] = min(settings["sourceDims"],srcDims)
    #XXX: determine memory and thread sizes from device
    settings["memSize"] = gpuMemSize*1024*1024
    settings["maxThreads"] = 1024
    
    #set up chunk sizes
    memoryPerElement = k*4*2+kmax*4*2 + (settings["sourceDims"]*4)*2 + 30 #+ (k+10)*4*2 #this is an estimated memory used per element
    MemoryPerElementSquared = 8*2
    ctr = 0
    while memoryPerElement*ctr + ctr*ctr*8*2 < settings["memSize"] and ctr*settings["sourceDims"]<1000000:
        ctr += 1
    ctr -= 1
    
    #settings = GPUConfig(settings,memoryPerElement,memoryPerElementsquared,1000000/settings["sourceDims"])
    
    settings["chunkSize"] = min(ctr,settings["dataLength"])
    settings["lastChunkSize"] = ((settings["dataLength"]-1) % settings["chunkSize"]) + 1
    
    #create kernel gridsize tuples
    settings["block"] = (settings["maxThreads"],1,1)
    settings["grid"] = ((settings["chunkSize"]/settings["maxThreads"])+(settings["chunkSize"]%settings["maxThreads"]>0),1,1)
    #precalculate all constant kernel params
    settings["dimensions"] = numpy.int64(settings["sourceDims"])
    settings["k"] = numpy.int64(k)
    settings["eps"] = numpy.float32(eps)
    settings["dataSize"] = numpy.int64(settings["dataLength"])
    settings["chunkSize"] = numpy.int64(settings["chunkSize"])
    settings["maxThreads"] = numpy.int64(settings["maxThreads"])
    
    return settings

def KNN(dataTable, k, epsilon=10000000000000., srcDims = 1000000000000000, normData = False):
    """
    Get a k,epsilon version k nearest neighbours
    """
    #load up the configuration
    knnOptions = KNNConfig(dataTable,srcDims,k,epsilon)
    
    
    #load and format the table for use.
    data = loadTable(dataTable,knnOptions)
    
    #check if we should normalise the data (this is really quick and dirty, replace it with something better)
    if normData:
        dmax = max([amax(d) for d in data])
        dmin = max([amin(d) for d in data])
        data = [(d-dmin)/(dmax-dmin+0.00000001) for d in data]
    
    #create the CUDA kernels
    header = "#define MAXKMAX ("+str(knnOptions['kmax'])+")\n#define CHUNKSIZE ("+str(knnOptions['chunkSize'])+")\n"
    program = SourceModule(header+open(KernelLocation+"KNN.nvcc").read())
    prg = program.get_function("KNN")
    sortprogram = SourceModule(header+open(KernelLocation+"KNNSORT.nvcc").read())
    sort = sortprogram.get_function("KNNSORT")
    t0 = time.time()
    
    #make a default distance list
    distances0 = (zeros((knnOptions['chunkSize']*knnOptions['k'])) + knnOptions['eps']).astype(numpy.float32)
    indices0 = zeros((knnOptions['chunkSize']*knnOptions['k'])).astype(numpy.uint32)
    dists = [distances0.copy() for i in xrange(len(data))]
    indices = [indices0.copy() for i in xrange(len(data))]
    
    
    #calculate KNN
    offset = 0
    source_gpu = drv.mem_alloc(data[0].nbytes)
    target_gpu = drv.mem_alloc(data[0].nbytes)
    indices_gpu = drv.mem_alloc(indices[0].nbytes)
    dists_gpu = drv.mem_alloc(dists[0].nbytes)
    
    data = [s.T.flatten() for s in data]
    for source in data:
        drv.memcpy_htod(source_gpu, source)
        drv.memcpy_htod(indices_gpu, indices[offset])
        drv.memcpy_htod(dists_gpu, dists[offset])
        for t in xrange(len(data)):
            drv.memcpy_htod(target_gpu, data[t])
            prg(source_gpu,
                target_gpu,
                indices_gpu,
                dists_gpu,
                knnOptions["dimensions"],
                knnOptions['k'],
                knnOptions['eps'],
                knnOptions['dataSize'],
                knnOptions['chunkSize'],
                numpy.int64(offset*knnOptions['chunkSize']),
                numpy.int64(t*knnOptions['chunkSize']),
                block=knnOptions['block'],
                grid=knnOptions['grid'])
        sort(dists_gpu,
            indices_gpu,
            numpy.int64(knnOptions['oldk']),
            knnOptions['dataSize'],
            knnOptions['chunkSize'],
            numpy.int64(offset*knnOptions['chunkSize']),
            block=knnOptions['block'],
            grid=knnOptions['grid'])
        drv.memcpy_dtoh(indices[offset], indices_gpu)
        drv.memcpy_dtoh(dists[offset], dists_gpu)
        offset += 1
    del source_gpu
    del indices_gpu
    del dists_gpu
    del target_gpu
    
    #organise data and add neighbours
    alldists = numpy.concatenate(dists).reshape((-1,knnOptions['k']))[:(knnOptions['dataSize']),:knnOptions['oldk']].tolist()
    allindices = numpy.concatenate(indices).reshape((-1,knnOptions['k']))[:(knnOptions['dataSize']),:knnOptions['oldk']].tolist()
    for i in xrange(len(alldists)): #remove excess entries
        if knnOptions['eps'] in alldists[i]:
            ind = alldists[i].index(knnOptions['eps'])
            alldists[i] = alldists[i][:ind]
            allindices[i] = allindices[i][:ind]
            ind = allindices[i].index(i)
            alldists[i] = alldists[i][:ind]
            allindices[i] = allindices[i][:ind]
    for i in xrange(len(alldists)):
            j = 0
            for ind in allindices[i]: #add mirrored entries
                if not (i in allindices[ind]):
                    allindices[ind].append(i)
                    alldists[ind].append(alldists[i][j])
                j += 1
    
    #have a list for start and end indices for knn
    klist = [0]
    for i in xrange(len(alldists)): #pad all entries to the same length for the next algorithm
        klist.append(len(alldists[i])+klist[-1])
    alldists = array([j for i in alldists for j in i],dtype=numpy.float32)
    allindices = array([j for i in allindices for j in i],dtype=numpy.uint32)
    klist = array(klist,dtype=numpy.uint32)
    
    print time.time()-t0, " seconds to process KNN"
    """
    #save the link matrix to compare to the matlab results
    f = open("linkmatrix.csv",'w')
    mat = [[0]*len(allindices) for i in xrange(len(allindices))]
    for i in xrange(len(allindices)):
        for j in xrange(len(allindices[i])):
            if alldists[i][j] < knnOptions['eps']:
                mat[i][allindices[i][j]] = mat[allindices[i][j]][i] = 1
    for m in mat:
        f.write(str(m).strip('[]')+'\n')
    f.close()
    """

    return allindices, alldists, klist #[allindices[i]+alldists[i] for i in xrange(len(alldists))]



