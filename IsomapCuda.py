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

#Normalisation for Eigen Embedding ---------------------------------------------

def NormMatrixConfig(dataTable, gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated APSP.
    """
    
    
    settings = dataConfig(dataTable,settings)
    
    settings["memSize"] = gpuMemSize*1024*1024
    settings["k"] = settings["sourceDims"]
    #settings["eps"] = eps
    
    #calculate memory constraints for the matrix chunks
    sumMem = 6*settings["dataLength"]
    chunkMem = (settings["memSize"]-sumMem)/4/settings["dataLength"]
    if chunkMem < 1:
        raise "GPU memory too small for norming the matrix!"
    
    settings["chunkSize"] = min(chunkMem,512)
    
    settings["lastChunkSize"] = settings["dataLength"] % settings["chunkSize"]
    
    if settings["lastChunkSize"] > 0:
        settings["totalChunks"] = settings["dataLength"]/settings["chunkSize"]+1
    else:
        settings["totalChunks"] = settings["dataLength"]/settings["chunkSize"]
    
    settings["block"] = (min(max(settings["dataLength"],1),512),1,1) #XXX: fix this for using the device data on max threads
    g1 = int(math.ceil(settings["dataLength"]/512.))
    g2 = int(math.ceil(g1/512.))
    g3 = int(math.ceil(g2/64.))
    settings["grid"] = (max(g1,1),max(g2,1),max(g3,1))
    return settings

def NormMatrix(dataTable,nmOptions):
        t0 = time.time()
        
        normedMatrix = loadMatrix(dataTable)
        
        normedMatrix = normedMatrix*normedMatrix
        normsums = array([sum(r) for r in normedMatrix])/nmOptions['dataLength']
        
        """
        progstr = ("__global__ void SumSquare(const unsigned int totalNodes, float* Sums, float* Paths) {\n"+
                   "    const unsigned int v = threadIdx.x+blockIdx.x*512+blockIdx.y*512*512;\n" +
                   "    if (v < "+str(nmOptions['dataLength'])+") {\n" +
                   "        double sum = 0.0;\n" +
                   "        for (unsigned int j = 0; j < "+str(nmOptions['chunkSize'])+"; j++) {\n" +
                   "            Paths[j*"+str(nmOptions['dataLength'])+"+v] = Paths[j*"+str(nmOptions['dataLength'])+"+v]*Paths[j*"+str(nmOptions['dataLength'])+"+v];\n" +
                   "            sum += Paths[j*"+str(nmOptions['dataLength'])+"+v];\n" +
                   "        }\n" +
                   "        Sums[v] += sum / "+str(nmOptions['dataLength'])+";\n" +
                   "    }\n" +
                   "}\n")
        program = SourceModule(progstr)
        prg = program.get_function("SumSquare")
        normsums = zeros(nmOptions['dataLength']).astype(numpy.float32)
        for i in xrange(nmOptions['totalChunks']):
            normsample = normedMatrix[i*nmOptions['chunkSize']:(i+1)*nmOptions['chunkSize']].flatten()
            prg(drv.In(numpy.uint32(nmOptions['dataLength'])),drv.InOut(normsums),drv.InOut(normsample),grid=nmOptions['grid'],block=nmOptions['block'])
            normedMatrix[i*nmOptions['chunkSize']:(i+1)*nmOptions['chunkSize']] = normsample.reshape((nmOptions['chunkSize'],nmOptions['dataLength']))
        """
        
        allsums = sum(normsums)/nmOptions['dataLength']
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

def C_Isomap(knndists,epsilon):
    mvals = []
    for k in knndists:
        ctr = 0
        while k[ctr] < epsilon:
            ctr += 1
        mvals.append(sum(k[:ctr])/ctr)
    return numpy.array(mvals,dtype=numpy.float32)
