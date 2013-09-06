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
GPU based approximate eigenvalue algorithm.
"""

import time
import numpy
import random
from numpy.linalg import eig
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import math


from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,loadIntMatrix,saveTable

KernelLocation = "CudaKernels/QEIG/"

# KMeans Algorithm --------------------------------------------

def QEConfig(dataTable, dims, k = -1, eps = 0.00001,gpuMemSize = 512, settings = {}):
    
    settings = dataConfig(dataTable,settings)
    
    #XXX: determine memory and thread sizes from device
    settings["memSize"] = gpuMemSize*1024*1024
    settings["maxThreads"] = 1024
    
    #set up chunk sizes - in this case degenerate
    settings["chunkSize"] = settings["dataLength"]
    
    #create kernel gridsize tuples
    settings["block"] = (settings["maxThreads"],1,1)
    settings["grid"] = (max(int(math.ceil(float(settings["chunkSize"])/settings["maxThreads"])),1),1,1)
    
    settings["k"] = k
    if k < 0:
        settings["k"] = min(settings["dataLength"]/125,50)
    settings["delta"] = eps
    settings["targetDims"] = dims
    
    return settings


def _calcRLS(dataMatrix,basisMatrix,basisSize,dataErrors):
    """
    Calculate the length squared orthonormalized error for all basis labels
    """
    
    #this calculates a slightly inaccurate residual of the orthonormal basis decomposition for all datapoints
    db = numpy.dot(dataMatrix,basisMatrix[:basisSize].T)
    
    for i in xrange(len(dataMatrix)):
        d = dataMatrix[i].copy()
        for j in xrange(basisSize):
            d -= basisMatrix[j]*db[i][j]
        dataErrors[i] = numpy.dot(d,d)**0.5

def _residual(dataVector,basisMatrix,basisSize,exclude=-1):
    """
    Calculate the length squared orthonormalized error for all basis labels
    """
    d = dataVector.copy()
    for i in xrange(basisSize):
        if not i == exclude:
            d -= numpy.dot(d,basisMatrix[i])*basisMatrix[i]
    return d

def QEig(dataTable,dims=3, k = -1, delta=0.00000002, initialBasis=[]):
    splitmax = 1
    
    qConfig = QEConfig(dataTable,dims,k,delta)
    t0 = time.time()
    
    basisMatrix = numpy.zeros((qConfig["k"]+splitmax,qConfig["sourceDims"])).astype(numpy.float32)
    dataLabels = numpy.zeros(qConfig["dataLength"]).astype(numpy.uint32)
    dataErrors = numpy.zeros(qConfig["dataLength"]).astype(numpy.float32)
    labelErrors = numpy.zeros(qConfig["k"]).astype(numpy.float32)
    
    #initialise tree
    basisSize = 1
    newBasis = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float64)
    for i in xrange(qConfig["dataLength"]):
        #"""
        nb = newBasis - dataTable[i]
        nb2 = newBasis + dataTable[i]
        if numpy.dot(nb,nb) > numpy.dot(nb2,nb2):
            newBasis = nb
        else:
            newBasis = nb2
        #"""
        #newBasis += dataTable[i]
    basisMatrix[0] = newBasis/numpy.sqrt(numpy.dot(newBasis,newBasis))
    labelData = [range(len(dataTable))]
    
    fnormData = sum(sum(dataTable*dataTable))**0.5
    
    RLS = fnormData
    while (RLS/fnormData)**2 > delta and basisSize < qConfig["k"]:
        #print basisSize, RLS,(RLS/fnormData)**2
        #Step 1: calculate data RLS
        _calcRLS(dataTable,basisMatrix,basisSize,dataErrors)
        
        #Step 2: sum RLS per label
        labelErrors *= 0.
        for i in xrange(len(dataErrors)):
            labelErrors[dataLabels[i]] += dataErrors[i]
        
        #Step 3: sort labels by RLS
        errorQueue = zip(labelErrors[:basisSize],range(basisSize))
        errorQueue.sort()
        largestLabel = errorQueue[-1][1]
        
        #Step 4: split label with largest RLS
        
        #4.1: make the split vector
        leaf = labelData[largestLabel]
        splitNode = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float64)
        for i in labelData[largestLabel]:
            nb = splitNode - dataTable[i]
            nb2 = splitNode + dataTable[i]
            if numpy.dot(nb,nb) > numpy.dot(nb2,nb2):
                splitNode = nb
            else:
                splitNode = nb2
        splitNode = _residual(splitNode,basisMatrix,basisSize,largestLabel)/labelErrors[largestLabel] #get the orthogonal decomposition error
        splitNode /= numpy.sqrt(numpy.dot(splitNode,splitNode)) + 0.0000000000000001 #normalise
        
        #4.2: calculate the cosines
        cosines = []
        for l in leaf:
            cosines.append(numpy.dot(dataTable[l],splitNode)/numpy.dot(dataTable[l],dataTable[l]))
        
        #4.3: sort the nodes
        cmax = max(cosines)
        cmin = min(cosines)
        left = []
        right = []
        
        for i in xrange(len(leaf)):
            if cmax-cosines[i] < cosines[i]-cmin:
                left.append(leaf[i])
            else:
                right.append(leaf[i])
        
        #4.4: relabel the nodes and increment the basis
        for r in right:
            dataLabels[r] = basisSize
        
        #4.5: recalculate the basis vectors
        
        #update left
        labelData[largestLabel] = left
        newBasis = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float64)
        for i in labelData[largestLabel]:
            nb = newBasis - dataTable[i]
            nb2 = newBasis + dataTable[i]
            if numpy.dot(nb,nb) > numpy.dot(nb2,nb2):
                newBasis = nb
            else:
                newBasis = nb2
        newBasis = _residual(newBasis,basisMatrix,basisSize,largestLabel) #get the orthogonal decomposition error
        newBasis /= numpy.sqrt(numpy.dot(newBasis,newBasis)) + 0.0000000000000001 #normalise
        basisMatrix[largestLabel] = newBasis
        
        
        #update right
        labelData.append(right)
        newBasis = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float64)
        for i in labelData[basisSize]:
            nb = newBasis - dataTable[i]
            nb2 = newBasis + dataTable[i]
            if numpy.dot(nb,nb) > numpy.dot(nb2,nb2):
                newBasis = nb
            else:
                newBasis = nb2
        newBasis = _residual(newBasis,basisMatrix,basisSize,basisSize) #get the orthogonal decomposition error
        newBasis /= numpy.sqrt(numpy.dot(newBasis,newBasis)) + 0.0000000000000001 #normalise
        basisMatrix[basisSize] = newBasis
        
        #increment basis
        basisSize += 1
        
        RLS = sum(labelErrors)/qConfig["dataLength"]
    
    X = numpy.dot(numpy.dot(basisMatrix[:basisSize],dataTable),basisMatrix[:basisSize].T)
    e,v = eig(X)
    v = v[:,:dims].real
    e = abs(e[:dims].real)**0.5
    
    print v.shape,e.shape,basisMatrix[:basisSize].shape
    
    out = (numpy.dot(basisMatrix[:basisSize].T,v).T) #[:dims]
    print time.time()-t0, " seconds to process Eigenvalues"
    print out.shape
    return out.T
    
    
    
