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

# APSP Algorithm ---------------------------------------------------
KernelLocation = "CudaKernels/APSP/"

def APSPConfig(dataTable, eps=100000000., gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated APSP.
    """
    
    settings = dataConfig(dataTable,settings)
    
    #XXX: determine memory and thread sizes from device
    settings["memSize"] = gpuMemSize*1024*1024
    settings["maxThreads"] = 1024
    
    #set up chunk sizes - in this case degenerate
    settings["chunkSize"] = settings["dataLength"]
    
    #create kernel gridsize tuples
    settings["block"] = (settings["maxThreads"],1,1)
    settings["grid"] = (max(int(math.ceil(float(settings["chunkSize"])/settings["maxThreads"])),1),1,1)
    
    #precalculate all constant kernel params
    settings["dimensions"] = numpy.int64(settings["sourceDims"])
    settings["k"] = numpy.int64(settings["sourceDims"])
    settings["eps"] = numpy.float32(eps)
    settings["dataSize"] = numpy.int64(settings["dataLength"])
    settings["chunkSize"] = numpy.int64(settings["chunkSize"])
    settings["maxThreads"] = numpy.int64(settings["maxThreads"])
    
    return settings
    


def APSP(knn_refs,knn_dists,eps):
    #timekeeping for profiling purposes
    t0 = time.time()
    
    apspOptions = APSPConfig(knn_dists,eps)
    
    #create the kernels
    sssp1 = SourceModule(open(KernelLocation+"SSSP.nvcc").read())
    kernel = sssp1.get_function("SSSP")
    seedprev = SourceModule(open(KernelLocation+"SEEDPREVIOUS.nvcc").read())
    seed = seedprev.get_function("SEEDPREVIOUS")
    
    #create our template cost list
    Costs0 = array([apspOptions['eps']]*apspOptions['dataLength']).astype(numpy.float32)
    Matrix = []
    
    
    #initialise our memory of how many iterations the previous row took to solve (used as a termination heuristic)
    last = 50
    
    #make a changed flag
    changed = numpy.zeros(1).astype(numpy.uint32)
    
    #initialise our GPU resident memory arrays
    refs_gpu = drv.mem_alloc(knn_refs.nbytes)
    dists_gpu = drv.mem_alloc(knn_dists.nbytes)
    costs1_gpu = drv.mem_alloc(Costs0.nbytes)
    costs2_gpu = drv.mem_alloc(Costs0.nbytes)
    
    changed_gpu = drv.mem_alloc(changed.nbytes)
    
    #copy static data onto the GPU
    drv.memcpy_htod(refs_gpu, knn_refs)
    drv.memcpy_htod(dists_gpu, knn_dists)
    
    #iterate through every row of the path cost matrix
    for v in xrange(0,apspOptions['dataLength']):
        
        #create a new row for the cost matrix, from the template list
        Costs = Costs0.copy().astype(numpy.float32)
        Costs[v] = 0.
        
        #initialise the costs we have for the immediate neighbours (this saves a single iteration of SSSP, and is faster to do this way)
        for n in xrange(apspOptions['k']):
            if knn_dists[v][n] < apspOptions['eps']:
                Costs[knn_refs[v][n]] = knn_dists[v][n]
        
        #copy the initial Costs row into the gpu
        drv.memcpy_htod(costs1_gpu, Costs)
        
        #pre-populate the costs with entries we've already solved
        #XXX: 10 is a hack, should be the degree of the vertex
        prefill = min(len(Matrix),10)
        for i in xrange(len(knn_refs[v])):
            s = knn_refs[v][i]
            if s < v and knn_dists[v][i] < apspOptions['eps']:
                drv.memcpy_htod(costs2_gpu, Matrix[s])
                seed(costs2_gpu, costs1_gpu,
                     numpy.int64(v),
                     apspOptions['dataSize'],
                     apspOptions['maxThreads'],
                     grid=apspOptions['grid'],
                     block=apspOptions['block'])
                prefill -=1
        if prefill > 0:
            for m in random.sample(Matrix,min(len(Matrix),prefill)):
                drv.memcpy_htod(costs2_gpu, m)
                seed(costs2_gpu, costs1_gpu,
                     numpy.int64(v),
                     apspOptions['dataSize'],
                     apspOptions['maxThreads'],
                     grid=apspOptions['grid'],
                     block=apspOptions['block'])
        
        
        #iteratively expand the shortest paths (1 iter per kernel run) until we have all the paths
        for i in xrange(last-2):
            #use 2 kernels copying back and forth between cost matrices to ensure no sync issues
            kernel( refs_gpu, dists_gpu,
                    costs1_gpu, costs2_gpu,
                    changed_gpu, apspOptions['dataSize'],
                    apspOptions['k'], apspOptions['eps'],
                    apspOptions['maxThreads'], 
                    grid=apspOptions['grid'],
                    block=apspOptions['block'])
            kernel( refs_gpu, dists_gpu,
                    costs2_gpu, costs1_gpu,
                    changed_gpu, apspOptions['dataSize'],
                    apspOptions['k'], apspOptions['eps'],
                    apspOptions['maxThreads'],
                    grid=apspOptions['grid'],
                    block=apspOptions['block'])
        l = last-12
        
        #XXX: is this the best way to pass a flag back?
        cval = 1
        changed[0] = 0
        drv.memcpy_htod(changed_gpu, changed)
        while cval != 0 or l > 1000:
            kernel( refs_gpu, dists_gpu,
                    costs1_gpu, costs2_gpu,
                    changed_gpu, apspOptions['dataSize'],
                    apspOptions['k'], apspOptions['eps'],
                    apspOptions['maxThreads'], 
                    grid=apspOptions['grid'],
                    block=apspOptions['block'])
            kernel( refs_gpu, dists_gpu,
                    costs2_gpu, costs1_gpu,
                    changed_gpu, apspOptions['dataSize'],
                    apspOptions['k'], apspOptions['eps'],
                    apspOptions['maxThreads'],
                    grid=apspOptions['grid'],
                    block=apspOptions['block'])
            l += 1
            drv.memcpy_dtoh(changed, changed_gpu)
            cval = changed[0]
            changed[0] = 0
            drv.memcpy_htod(changed_gpu, changed)
            
        last = max(l,1)
        
        #copy costs back into memory
        drv.memcpy_dtoh(Costs, costs1_gpu)
        
        #add the row to the matrix
        Matrix.append(Costs)
    
    #explicitly free gpu memory
    del costs1_gpu
    del costs2_gpu
    del refs_gpu
    del dists_gpu
    
    print time.time()-t0, " seconds to create shortest paths."
    
    """
    #save out the matrix to compare to the matlab reference version
    f = open('pathmatrix.csv','w')
    for line in Matrix:
        f.write(str(list(line)).strip('[]')+'\n')
    f.close()
    """
    
    return array(Matrix)

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


#Get ND Eigen Embedding --------------------------------------------------------

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
    
    return array(result)[:,1:].astype(numpy.uint32) #,array(result)[:,:1].astype(numpy.float32)

