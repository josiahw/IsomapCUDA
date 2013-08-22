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
import numpy
from numpy.linalg import eig
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math


from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,loadIntMatrix,saveTable
from KNearestNeighbours import KNN
from NonMetricMultiDimensionalScaling import NMDS

# APSP Algorithm ---------------------------------------------------

def APSPConfig(dataTable, eps=100000000., gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated APSP.
    """
    
    
    settings = dataConfig(dataTable,settings)
    settings["sourceDims"] /= 2 #the data has indices and values on the same line this time
    
    settings["memSize"] = gpuMemSize*1024*1024
    settings["k"] = settings["sourceDims"]
    settings["eps"] = eps
    
    #calculate memory constraints for the KNN chunks
    knnMem = settings["k"]*(4+4)*settings["dataLength"] #we have ( k * size( float + int ) * dataLength ) for our neighbour list
    chunkMem = settings["memSize"]-knnMem
    if chunkMem < settings["sourceDims"]*4*4:
        raise "GPU memory too small for KNN list!"
    chunkMem /= settings["sourceDims"]*4*4
    
    settings["chunkSize"] = 1 #min(chunkMem,512)
    
    settings["lastChunkSize"] = settings["dataLength"] % settings["chunkSize"]
    
    if settings["lastChunkSize"] > 0:
        settings["totalChunks"] = settings["dataLength"]/settings["chunkSize"]+1
    else:
        settings["totalChunks"] = settings["dataLength"]/settings["chunkSize"]
    
    t_threads = 1024
    settings["t_threads"] = t_threads
    settings["block"] = (min(max(settings["dataLength"],1),t_threads),1,1) #XXX: fix this for using the device data on max threads
    g1 = int(math.ceil(settings["dataLength"]/float(t_threads)))
    g2 = int(math.ceil(g1/float(t_threads)))
    g3 = int(math.ceil(g2/64.))
    settings["grid"] = (max(g1,1),max(g2,1),max(g3,1))
    return settings
    

def APSPKernel(targetChunkSize,options,prefix = ''):
    """
    Return the string representation of the desired tiled KNN kernel.
    We do this twice for different target chunk sizes so we can handle odd length data.
    """
    #choose a multiple of 2 for unrolling
    unroll = 1
    
    
    #implemented from paper
    progstr = ("__global__ void SSSP(const unsigned int* Edges, const float* Weights, const float* Costs, float* Paths) {\n"+
               "    const unsigned int v = threadIdx.x+blockIdx.x*"+str(options["t_threads"])+"+blockIdx.y*"+str(options["t_threads"])+"*"+str(options["t_threads"])+";\n" +
               "    if (v < "+str(options['dataLength'])+") {\n" +
               "        const unsigned int vertex = v;\n" +
               "        float p = Costs[vertex];\n" +
               "        unsigned int i = 0;\n" +
               "        while (Weights[vertex*"+str(options['k'])+"+i] < "+str(options['eps'])+" and i < "+str(options['k'])+") {\n" +
               "            const unsigned int neighbourid = vertex*"+str(options['k'])+"+i;\n" +
               "            const double d = (Costs[Edges[neighbourid]]+Weights[neighbourid]);\n" +
               "            p = min(p,d);\n" +
               "            i++;\n" +
               "        }\n"+
               "        Paths[vertex] = p;\n" +
               "    }\n" +
               "}\n")
    #print progstr
    return progstr


def APSP(dataTable,apspOptions):
    knn_refs,knn_dists = loadSplitTable(dataTable,apspOptions)
    knn_refs = knn_refs.astype(numpy.uint32)
    knn_dists = knn_dists.astype(numpy.float32)
    
    #neighbours = numpy.int32(self.number_of_neighbours)
    
    sssp1 = SourceModule(APSPKernel(apspOptions['chunkSize'],apspOptions))
    kernel = sssp1.get_function("SSSP")
    
    #print str(self.data_length/self.num_threads),self.data_length/self.num_threads*200
    
    Costs0 = array([apspOptions['eps']]*apspOptions['dataLength']).astype(numpy.float32)
    Matrix = []
    t0 = time.time()
    
    #iterate through every row of the path cost matrix
    last = 70
    for v in xrange(0,apspOptions['dataLength'],apspOptions['chunkSize']):
        
        #create a new row for the cost matrix
        Costs = Costs0.copy().astype(numpy.float32)
        
        Costs[v] = 0.
        for n in xrange(v):
            Costs[n] = Matrix[n][v]
        v2 = v*apspOptions['k']
        
        #initialise the costs we have for the immediate neighbours
        for n in xrange(apspOptions['k']):
            if knn_dists[v2+n] < apspOptions['eps']:
                #print v2+n,knn_dists[v2+n]
                Costs[knn_refs[v2+n]] = knn_dists[v2+n]
        
        Costs2 = Costs.copy().astype(numpy.float32)
        
        #iteratively expand the shortest paths (1 iter per kernel run) until we have all the paths
        for i in xrange(last-3):
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs),drv.Out(Costs2),grid=apspOptions['grid'],block=apspOptions['block'])
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs2),drv.Out(Costs),grid=apspOptions['grid'],block=apspOptions['block'])
        l = last-3
        
        
        #XXX: this is expensive, find a better way
        while amax(Costs) > 100000.:
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs),drv.Out(Costs2),grid=apspOptions['grid'],block=apspOptions['block'])
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs2),drv.Out(Costs),grid=apspOptions['grid'],block=apspOptions['block'])
            l += 1
            #print Costs
            
        last = l
        
        
        #do a few extra iterations in case we missed the shortest paths
        for i in xrange(15):
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs),drv.Out(Costs2),block=apspOptions['block'])
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs2),drv.Out(Costs),block=apspOptions['block'])
        
        #print Costs
        
        #copy our paths to the diagonally reflected side of the matrix, to guarantee symmetry
        for n in xrange(v):
            Matrix[n][v] = Costs[n]
        
        
        #add the row to the matrix
        Matrix.append(Costs)
        
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
"""
#XXX: need global ranks, not ranks per index
def RankMatrix(dataTable):
    t0 = time.time()
    
    result = []
    
    for l in dataTable:
        sl = zip(l,range(len(l)))
        sl.sort()
        result.append(array(zip(*sl)[1]))
    
    print time.time()-t0, " seconds to compute Rank Matrix."
    
    return array(result)
"""
def RankMatrix(dataTable):
    t0 = time.time()
    result = []
    for i in xrange(0,len(dataTable)):
        for j in xrange(i+1,len(dataTable)):
            result.append((dataTable[i][j],i,j))
    result.sort()
    print time.time()-t0, " seconds to compute Rank Matrix."
    
    return array(result)[:,1:].astype(numpy.uint32) #,array(result)[:,:1].astype(numpy.float32)

