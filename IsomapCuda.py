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
GPU and CPU based K nearest neighbours algorithms.
"""

import time
from numpy import array,zeros,amax,amin,sqrt,dot,random
import scipy
import numpy
from numpy.linalg import eig
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math
import random


from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,loadIntMatrix,saveTable
from KNearestNeighbours import KNN
from NonMetricMultiDimensionalScaling import NMDS
from QuicEig import QEig
from ShortestPaths import APSP

#Get RankMatrix --------------------------------------------------------

def RankMatrix(dataTable):
    t0 = time.time()
    result = []
    for i in xrange(0,len(dataTable)):
        for j in xrange(i+1,len(dataTable)):
            result.append((dataTable[i][j],i,j))
    result.sort()
    print time.time()-t0, " seconds to compute Rank Matrix."
    
    return numpy.array(result)[:,1:].astype(numpy.uint32) #,array(result)[:,:1].astype(numpy.float32)


#Get C-Isomap m-values --------------------------------------------------------

def C_Isomap(knndists,knnm,k):
    mvals = zeros(len(knnm)-1,dtype=numpy.float32)
    for i in xrange(len(knnm)-1):
        mvals[i] = (sum(knndists[knnm[i]:knnm[i]+k])/k)
    return numpy.sqrt(mvals)

#Normalisation for Eigen Embedding ---------------------------------------------

def NormMatrix(dataTable, mlabels = []):
        dataLength = len(dataTable)
        t0 = time.time()
        
        normedMatrix = dataTable*dataTable
        if len(mlabels):
            for i in xrange(len(normedMatrix)):
                normedMatrix[i] /=mlabels[i]
                normedMatrix[:,i] /=mlabels[i]
        
        normsums = numpy.sum(normedMatrix,axis=1)/dataLength
        
        allsums = numpy.sum(normsums)/dataLength
        for i in xrange(len(normedMatrix)):
            normedMatrix[i] = -0.5*(normedMatrix[i] - normsums - normsums[i] + allsums)
        print time.time()-t0, " seconds to normalize the matrix."
        
        """
        #export the matrix to compare against the matlab implementation
        f = open("normedmatrix.csv",'w')
        for l in normedMatrix:
            f.write(str(list(l)).strip('[]')+'\n')
        f.close()
        """
        
        return normedMatrix


#Get exact ND Eigen Embedding --------------------------------------------------------

def EigenEmbedding(dataTable, finalDims = 3):
    t0 = time.time()
        
    e = eig(loadMatrix(dataTable))
    e = [(e[1].real.T[i]*sqrt(abs(e[0].real[i]))).tolist() for i in xrange(len(e[0]))]
    #e.reverse()
    e = [list(l) for l in zip(*e[:finalDims])]
    
    print time.time()-t0, " seconds to compute eigenvalue embedding."
    
    return e


    
