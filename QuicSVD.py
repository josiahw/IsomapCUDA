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
GPU based approximate SVD algorithm.
"""

import time
import numpy
import random
from numpy.linalg import svd
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math


from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,loadIntMatrix,saveTable

KernelLocation = "CudaKernels/QSVD/"

# KMeans Algorithm --------------------------------------------

def QSVDConfig(dataTable, dims, k = -1, eps = 0.00001,gpuMemSize = 512, settings = {}):
    
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
        settings["k"] = settings["dataLength"]/20
    settings["delta"] = eps
    settings["targetDims"] = dims
    
    return settings


def _calcRLS(dataMatrix,basisMatrix,dataLabels,dataErrors):
    """
    Calculate the length squared orthonormalized error for all basis labels
    """
    for i in xrange(len(dataMatrix)):
        d = dataMatrix[i].copy()
        for b in basisMatrix:
            d -= numpy.dot(d,b)*b
        dataErrors[i] = numpy.dot(d,d)

def _fnorm2(basisMatrix,basisSize):
    """
    Calculates the frobenius norm squared
    """
    return sum(sum(basisMatrix[:basisSize]*basisMatrix[:basisSize]))

def _residual(dataVector,basisMatrix,exclude=-1):
    """
    Calculate the length squared orthonormalized error for all basis labels
    """
    d = dataVector.copy()
    for i in xrange(len(basisMatrix)):
        if not i == exclude:
            d -= numpy.dot(d,basisMatrix[i])*basisMatrix[i]
    return d

def QSVD(dataTable,dims=3, k = -1, delta=0.00001, initialBasis=[]):
    
    
    qConfig = QSVDConfig(dataTable,dims,k,delta)
    
    basisMatrix = numpy.zeros((qConfig["k"],qConfig["sourceDims"])).astype(numpy.float32)
    dataLabels = numpy.zeros(qConfig["dataLength"]).astype(numpy.uint32)
    dataErrors = numpy.zeros(qConfig["dataLength"]).astype(numpy.float32)
    labelErrors = numpy.zeros(qConfig["k"]).astype(numpy.float32)
    
    #initialise tree
    basisSize = 1
    newBasis = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float32)
    for i in xrange(qConfig["dataLength"]):
        newBasis += dataTable[i]
    basisMatrix[0] = newBasis/numpy.sqrt(numpy.dot(newBasis,newBasis))
    
    RLS = 1.
    while RLS > delta and basisSize < qConfig["k"]:
        print "iterating ",RLS
        #Step 1: calculate data RLS
        _calcRLS(dataTable,basisMatrix,dataLabels,dataErrors)
        
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
        leaf = []
        splitNode = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float32)
        for i in xrange(qConfig["dataLength"]):
            if dataLabels[i] == largestLabel:
                splitNode += dataTable[i]*dataErrors[i] #get the orthogonal decomposition
                leaf.append(i)
        splitNode = _residual(splitNode,basisMatrix)/labelErrors[largestLabel] #get the orthogonal decomposition error
        splitNode /= numpy.sqrt(numpy.dot(splitNode,splitNode)) #normalise
        
        #4.2: calculate the cosines
        cosines = []
        for l in leaf:
            cosines.append(numpy.dot(dataTable[l],splitNode)/numpy.sqrt(numpy.dot(dataTable[l],dataTable[l])))
        
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
        newBasis = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float32)
        for i in xrange(qConfig["dataLength"]):
            if dataLabels[i] == largestLabel:
                newBasis += dataTable[i] #get the orthogonal decomposition
        newBasis = _residual(newBasis,basisMatrix,largestLabel) #get the orthogonal decomposition error
        newBasis /= numpy.sqrt(numpy.dot(newBasis,newBasis)) #normalise
        basisMatrix[largestLabel] = newBasis
        
        #update right
        newBasis = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float32)
        for i in xrange(qConfig["dataLength"]):
            if dataLabels[i] == largestLabel:
                newBasis += dataTable[i] #get the orthogonal decomposition
        newBasis = _residual(newBasis,basisMatrix,basisSize) #get the orthogonal decomposition error
        newBasis /= numpy.sqrt(numpy.dot(newBasis,newBasis)) #normalise
        basisMatrix[basisSize] = newBasis
        
        #increment basis
        basisSize += 1
        
        RLS = sum(labelErrors)/qConfig["dataLength"]
        
    #Now perform SVD of our subspace basis:
    X = numpy.dot(dataTable,basisMatrix[:basisSize].T)
    u,s,v = svd(numpy.dot(X.T,X))
    
    #print s
    s2 = 1./s
    
    u = numpy.dot(numpy.dot(X,v),s2) #we only need u
    #for i in xrange(len(s)):
    #    s[i][i] = numpy.sqrt(s[i][i])
    #v = numpy.dot(basisMatrix[:basisSize].T,v)
    
    print u[0]
    out = []
    for i in xrange(qConfig["targetDims"]):
        eig = numpy.dot(u[i],dataTable)
        eig = numpy.dot(eig,u[i])/numpy.dot(u[i],u[i]) #calculate eigenvalue
        
        out.append(numpy.sqrt(eig)*u[i])
    #print out[0]
    return numpy.array(out)
    
    
    
