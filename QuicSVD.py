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
from numpy.linalg import svd,eig
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
        settings["k"] = settings["dataLength"]/50
    settings["delta"] = eps
    settings["targetDims"] = dims
    
    return settings


def _calcRLS(dataMatrix,basisMatrix,basisSize,dataErrors):
    """
    Calculate the length squared orthonormalized error for all basis labels
    """
    for i in xrange(len(dataMatrix)):
        d = dataMatrix[i].copy()
        for b in basisMatrix[:basisSize]:
            d -= numpy.dot(d,b)*b
        dataErrors[i] = numpy.dot(d,d)

def _fnorm2(basisMatrix,basisSize):
    """
    Calculates the frobenius norm squared
    """
    return sum(sum(basisMatrix[:basisSize]*basisMatrix[:basisSize]))

def _residual(dataVector,basisMatrix,basisSize,exclude=-1):
    """
    Calculate the length squared orthonormalized error for all basis labels
    """
    d = dataVector.copy()
    for i in xrange(basisSize):
        if not i == exclude:
            d -= numpy.dot(d,basisMatrix[i])*basisMatrix[i]
    return d

def QSVD(dataTable,dims=3, k = -1, delta=0.00001, initialBasis=[]):
    splitmax = 1
    
    qConfig = QSVDConfig(dataTable,dims,k,delta)
    
    
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
        #ewBasis += dataTable[i]
    basisMatrix[0] = newBasis/numpy.sqrt(numpy.dot(newBasis,newBasis))
    
    
    fnormData = sum(sum(dataTable*dataTable))
    
    RLS = 1.
    while RLS > delta and basisSize < qConfig["k"]:
        print basisSize, RLS,RLS/fnormData
        #Step 1: calculate data RLS
        _calcRLS(dataTable,basisMatrix,basisSize,dataErrors)
        
        #Step 2: sum RLS per label
        labelErrors *= 0.
        for i in xrange(len(dataErrors)):
            labelErrors[dataLabels[i]] += dataErrors[i]
        
        #Step 3: sort labels by RLS
        errorQueue = zip(labelErrors[:basisSize],range(basisSize))
        errorQueue.sort()
        for ii in xrange(min(splitmax,len(errorQueue))):
            largestLabel = errorQueue[-1][1]
            errorQueue.pop()
            
            #Step 4: split label with largest RLS
            
            #4.1: make the split vector
            leaf = []
            splitNode = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float64)
            for i in xrange(qConfig["dataLength"]):
                if dataLabels[i] == largestLabel:
                    """
                    nb = splitNode - dataTable[i]
                    nb2 = splitNode + dataTable[i]
                    if numpy.dot(nb,nb) > numpy.dot(nb2,nb2):
                        splitNode = nb
                    else:
                        splitNode = nb2
                    """
                    splitNode += dataTable[i]*dataErrors[i] #*numpy.dot(dataTable[i],dataTable[i]) # #get the orthogonal decomposition
                    leaf.append(i)
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
            newBasis = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float64)
            for i in xrange(qConfig["dataLength"]):
                if dataLabels[i] == largestLabel:
                    #"""
                    nb = newBasis - dataTable[i]
                    nb2 = newBasis + dataTable[i]
                    if numpy.dot(nb,nb) > numpy.dot(nb2,nb2):
                        newBasis = nb
                    else:
                        newBasis = nb2
                    #"""
                    #newBasis += dataTable[i] #*dataErrors[i] #get the orthogonal decomposition
            newBasis = _residual(newBasis,basisMatrix,basisSize,largestLabel) #get the orthogonal decomposition error
            newBasis /= numpy.sqrt(numpy.dot(newBasis,newBasis)) + 0.0000000000000001 #normalise
            basisMatrix[largestLabel] = newBasis
            
            #update right
            newBasis = numpy.zeros(qConfig["sourceDims"]).astype(numpy.float64)
            for i in xrange(qConfig["dataLength"]):
                if dataLabels[i] == basisSize:
                    #"""
                    nb = newBasis - dataTable[i]
                    nb2 = newBasis + dataTable[i]
                    if numpy.dot(nb,nb) > numpy.dot(nb2,nb2):
                        newBasis = nb
                    else:
                        newBasis = nb2
                    #"""
                    #newBasis += dataTable[i] #*dataErrors[i] #get the orthogonal decomposition
            newBasis = _residual(newBasis,basisMatrix,basisSize,basisSize) #get the orthogonal decomposition error
            newBasis /= numpy.sqrt(numpy.dot(newBasis,newBasis)) + 0.0000000000000001 #normalise
            basisMatrix[basisSize] = newBasis
            
            #increment basis
            basisSize += 1
        
        RLS = sum(labelErrors)/qConfig["dataLength"]
    
    
    """
    #OLD SVD CODE - NOT SURE IF THIS COULD BE FASTER THAN EIGS
    #basisSize = len(dataTable)-500
    #Now perform SVD of our subspace basis:
    X = numpy.dot(dataTable,basisMatrix[:basisSize].T)
    #u,s,v = svd(numpy.dot(X.T,X))
    u,s,v = svd(numpy.dot(basisMatrix[:basisSize],X))
    print X.shape
    print basisMatrix[:basisSize].shape
    print u.shape,s.shape,v.shape
    #print s
    #s2 = 1./s**0.5
    
    
    #u = numpy.dot(numpy.dot(v,basisMatrix[:basisSize]),dataTable) #we only need u
    u = numpy.dot(numpy.dot(basisMatrix[:basisSize].T,v.T),v)
    for i in xrange(len(u)):
        u[i] /= numpy.dot(u[i],u[i])**0.5 #s2
    u = numpy.dot(u.T,dataTable)
    #for i in xrange(len(s)):
    #    s[i][i] = numpy.sqrt(s[i][i])
    #v = numpy.dot(basisMatrix[:basisSize].T,v)
    print u.shape
    out = []
    for i in xrange(0,qConfig["targetDims"]):
        
        eig = numpy.dot(u[i],dataTable)
        eig = numpy.dot(eig,u[i]) #/(numpy.dot(u[i],u[i])+0.000000000000001) #calculate eigenvalue
        #print numpy.dot(u[i],u[i]),eig
        out.append(eig**0.5*u[i]) #/numpy.dot(u[i],u[i]))
    print out[0]
    """
    X = numpy.dot(numpy.dot(basisMatrix[:basisSize],dataTable),basisMatrix[:basisSize].T)
    e,v = eig(X)
    v = v.real
    e = abs(e.real)**0.5
    
    out = (e*numpy.dot(basisMatrix[:basisSize].T,v)).T
    
    
    return out.T
    
    
    
