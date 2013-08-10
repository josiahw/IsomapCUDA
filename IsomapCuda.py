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
Set of (mostly) GPU based algorithms for doing Isomap and Isomap variants
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


# KNN Algorithm --------------------------------------------

def simpleKNN(dataTable, k, eps = 1000000000):
    """
    A simple CPU implementation to check KNN against
    """
    data = loadMatrix(dataTable)
    
    allindices = []
    alldists = []
    for d in data:
        distances = [sqrt(dot(d-t,d-t)) for t in data]
        
        p = zip(distances,range(len(distances)))
        p.sort()
        p = p[1:k+1]
        dists,inds = zip(*p)
        dists = list(dists)
        inds = list(inds)
        
        alldists.append(dists)
        allindices.append(inds)
    for i in xrange(len(alldists)):
            j = 0
            for ind in allindices[i]: #add mirrored entries
                if not (i in allindices[ind]):
                    allindices[ind].append(i)
                    alldists[ind].append(alldists[i][j])
                j += 1
    
    maxKValues = max([len(p) for p in allindices]) 
    #print maxKValues
    for i in xrange(len(alldists)): #pad all entries to the same length for the next algorithm
        if len(alldists[i]) < maxKValues:
            alldists[i].extend( [eps]*(maxKValues-len(alldists[i])) )
            allindices[i].extend( [0]*(maxKValues-len(allindices[i])) )
    return [allindices[i]+alldists[i] for i in xrange(len(alldists))]

def KNNConfig(dataTable, k, eps = 1000000000.,gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated KNN.
    """
    settings = dataConfig(dataTable,settings)
    
    settings["memSize"] = gpuMemSize*1024*1024
    settings["k"] = k
    settings["eps"] = eps
    
    #calculate memory constraints for the KNN chunks
    knnMem = k*(4+4)*settings["dataLength"] #we have ( k * size( float + int ) * dataLength ) for our neighbour list
    chunkMem = settings["memSize"]-knnMem
    if chunkMem < settings["sourceDims"]*4*3:
        raise "GPU memory too small for KNN list!"
    chunkMem /= settings["sourceDims"]*4*3
    
    settings["chunkSize"] = min(chunkMem,512)
    
    settings["lastChunkSize"] = settings["dataLength"] % settings["chunkSize"]
    
    if settings["lastChunkSize"] > 0:
        settings["totalChunks"] = settings["dataLength"]/settings["chunkSize"]+1
    else:
        settings["totalChunks"] = settings["dataLength"]/settings["chunkSize"]
    return settings

def KNNKernel(targetChunkSize,options,prefix = ''):
    """
    Return the string representation of the desired tiled KNN kernel.
    We do this twice for different target chunk sizes so we can handle odd length data.
    """
    
    program =  ("__global__ void "+prefix+"knn(const float* source, const float* target, float* distances, unsigned int* indices, const long target_offset, const long source_offset) {\n"+
                "    const unsigned int source_begin = (threadIdx.x)*"+str(options['sourceDims'])+";\n"+
                "    const unsigned int source_address = (threadIdx.x)*"+str(options['k'])+";\n"+
                "    const unsigned int target_index = (target_offset);\n"+
                "    \n"+
                "    for (unsigned int i = 0; i < "+str(targetChunkSize)+"; i++) {\n"+
                "        double distance = 0.;\n"+
                "        for (unsigned int j = 0; j < "+str(options['sourceDims'])+"; j++) {\n"+
                "            distance += ((target[i*"+str(options['sourceDims'])+"+j]-source[source_begin+j])*\n"+
                "                        (target[i*"+str(options['sourceDims'])+"+j]-source[source_begin+j]));\n"+
                "        }\n"+
                "        distance = sqrt(distance);\n"+
                "        if (distance <= distances[source_address+"+str(options['k'])+"-1] and distance < "+str(options['eps'])+" and (i+target_offset) != (threadIdx.x+source_offset)) {\n"+
                "            unsigned int j = 0;\n"+
                "            while (distance > distances[source_address+j]) {\n"+
                "                j++;\n"+
                "            }\n"+
                "            for (unsigned int k = "+str(options['k'])+"-2; k >= j; k--) {\n"+
                "                distances[source_address+k+1] = distances[source_address+k];\n"+
                "                indices[source_address+k+1] = indices[source_address+k];\n"+
                "            }\n"+
                "            distances[source_address+j] = distance;\n"+
                "            indices[source_address+j] = target_index+i;\n"+
                "        }\n"+
                "    }\n" +
                "}\n")
    return program

def KNN(dataTable,knnOptions):
    """
    Get a k,epsilon version k nearest neighbours
    """
    #load and format the table for use.
    data = loadTable(dataTable,knnOptions)
    
    #create the CUDA kernels
    program = SourceModule(KNNKernel(knnOptions['chunkSize'],knnOptions))
    prg = program.get_function("knn")
    program2 = SourceModule(KNNKernel(knnOptions['lastChunkSize'],knnOptions,'last'))
    prg2 = program2.get_function("lastknn")
    t0 = time.time()
    
    #make a default distance list
    distances0 = (zeros((knnOptions['chunkSize']*knnOptions['k'])) + knnOptions['eps']).astype(numpy.float32)
    indices0 = zeros((knnOptions['chunkSize']*knnOptions['k'])).astype(numpy.uint32)
    dists = [distances0.copy() for i in xrange(knnOptions['totalChunks'])]
    indices = [indices0.copy() for i in xrange(knnOptions['totalChunks'])]
    
    #calculate KNN
    for t in xrange(len(data)-1):
        offset2 = 0
        for source in data[:-1]:
            prg(drv.In(source),drv.In(data[t]),drv.InOut(dists[offset2]),drv.InOut(indices[offset2]),numpy.int64(t*knnOptions['chunkSize']),numpy.int64(offset2*knnOptions['chunkSize']),block=(knnOptions['chunkSize'],1,1))
            offset2 += 1
        prg(drv.In(data[-1]),drv.In(data[t]),drv.InOut(dists[-1]),drv.InOut(indices[-1]),numpy.int64(t*knnOptions['chunkSize']),numpy.int64(offset2*knnOptions['chunkSize']),block=(knnOptions['lastChunkSize'],1,1))
    t = knnOptions['totalChunks']-1
    offset2 = 0
    for source in data[:-1]:
        prg2(drv.In(source),drv.In(data[t]),drv.InOut(dists[offset2]),drv.InOut(indices[offset2]),numpy.int64(t*knnOptions['chunkSize']),numpy.int64(offset2*knnOptions['chunkSize']),block=(knnOptions['chunkSize'],1,1))
        offset2 += 1
    prg2(drv.In(data[-1]),drv.In(data[t]),drv.InOut(dists[-1]),drv.InOut(indices[-1]),numpy.int64(t*knnOptions['chunkSize']),numpy.int64(offset2*knnOptions['chunkSize']),block=(knnOptions['lastChunkSize'],1,1))
    
    #organise data and add neighbours
    alldists = []
    allindices = []
    for i in xrange(knnOptions['totalChunks']): #rearrange into single lists
        alldists += dists[i].reshape((knnOptions['chunkSize'],knnOptions['k'])).tolist()
        allindices += indices[i].reshape((knnOptions['chunkSize'],knnOptions['k'])).tolist()
    alldists = alldists[:-(knnOptions['chunkSize']-knnOptions['lastChunkSize'])] #.tolist()
    allindices = allindices[:-(knnOptions['chunkSize']-knnOptions['lastChunkSize'])] #.tolist()
    for i in xrange(len(alldists)): #remove excess entries
        if knnOptions['eps'] in alldists[i]:
            ind = alldists[i].index(knnOptions['eps'])
            alldists[i] = alldists[i][:ind]
            allindices[i] = allindices[i][:ind]
    for i in xrange(len(alldists)):
            j = 0
            for ind in allindices[i]: #add mirrored entries
                if not (i in allindices[ind]):
                    allindices[ind].append(i)
                    alldists[ind].append(alldists[i][j])
                j += 1
    
    maxKValues = max([len(p) for p in allindices]) 
    #print maxKValues
    for i in xrange(len(alldists)): #pad all entries to the same length for the next algorithm
        if len(alldists[i]) < maxKValues:
            alldists[i].extend( [knnOptions['eps']]*(maxKValues-len(alldists[i])) )
            allindices[i].extend( [0]*(maxKValues-len(allindices[i])) )
    
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

    return [allindices[i]+alldists[i] for i in xrange(len(alldists))]



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
    
    settings["block"] = (min(max(settings["dataLength"],1),512),1,1) #XXX: fix this for using the device data on max threads
    g1 = int(math.ceil(settings["dataLength"]/512.))
    g2 = int(math.ceil(g1/512.))
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
               "    const unsigned int v = threadIdx.x+blockIdx.x*512+blockIdx.y*512*512;\n" +
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
    
    return array(result)[:,1:].astype(numpy.uint32)

#Non-metric MDS algorithm -------------------------------------------------------
def CPUNMDS1(dataTable,origData,nmdsOptions):
    rank_matrix = loadIntMatrix(dataTable)
    numchunks = 10 #len(rank_matrix)/200000
    chunksize = len(rank_matrix)/numchunks
    
    rank_matrix = [rank_matrix[i*chunksize:(i+1)*chunksize].astype(numpy.uint32) for i in xrange(numchunks)]
    t0 = time.time()
    
    #prepare an embedding and adjustments
    embeddingCoords = array(origData)[:,:nmdsOptions['targetDims']].astype(numpy.float32) #random.normal(0.,1.,(nmdsOptions['sourceDims'],nmdsOptions['targetDims'])) #.astype(numpy.float32)
    threads = 1024
    KernelHeader = ('#define DATA_SIZE ('+str(nmdsOptions['sourceDims'])+')\n'+
                    '#define TOTAL_THREADS ('+str(threads)+')\n'+
                    '#define DATA_STEP_SIZE ('+str(int(math.ceil(nmdsOptions['sourceDims']/float(threads))))+')\n'+
                    '#define DATA_LENGTH ('+str(nmdsOptions['sourceDims'])+')\n'+
                    '#define DATA_DIMS ('+str(nmdsOptions['targetDims'])+')\n'+
                    '#define ALPHA (0.15)\n')
    LastKernelHeader = ('#define CHUNK_SIZE ('+str(len(rank_matrix[-1]))+')\n'+
                             '#define STEP_SIZE ('+str(int(math.ceil(len(rank_matrix[-1])/float(threads))))+')\n'+
                             KernelHeader)
    KernelHeader = ('#define CHUNK_SIZE ('+str(chunksize)+')\n'+
                    '#define STEP_SIZE ('+str(int(math.ceil(chunksize/float(threads))))+')\n'+
                    KernelHeader)
    
    PAVkernel = KernelHeader + open("PAV.nvcc").read()
    DISTkernel = KernelHeader + open("RANKDIST.nvcc").read()
    DELTAkernel = KernelHeader + open("NMDS.nvcc").read()
    SCALEkernel = KernelHeader + open("SCALE.nvcc").read()
    lastPAVkernel = LastKernelHeader + open("PAV.nvcc").read()
    lastDISTkernel = LastKernelHeader + open("RANKDIST.nvcc").read()
    lastDELTAkernel = LastKernelHeader + open("NMDS.nvcc").read()
    lastSCALEkernel = LastKernelHeader + open("SCALE.nvcc").read()
    pk = SourceModule(PAVkernel)
    dk = SourceModule(DISTkernel)
    ek = SourceModule(DELTAkernel)
    sk = SourceModule(SCALEkernel)
    kernel = pk.get_function("PAV")
    distkernel = dk.get_function("RankDist")
    deltakernel = ek.get_function("NMDS")
    scalekernel = sk.get_function("Scale")
    lpk = SourceModule(lastPAVkernel)
    ldk = SourceModule(lastDISTkernel)
    lek = SourceModule(lastDELTAkernel)
    lsk = SourceModule(SCALEkernel)
    lkernel = lpk.get_function("PAV")
    ldistkernel = ldk.get_function("RankDist")
    ldeltakernel = lek.get_function("NMDS")
    lscalekernel = lsk.get_function("Scale")
    
    
    d = [zeros(chunksize).astype(numpy.float32) for n in xrange(numchunks)]
    od = [zeros(chunksize).astype(numpy.float32) for n in xrange(numchunks)]
    sums = zeros(threads).astype(numpy.float32)
    #diffs = zeros((len(rank_matrix),nmdsOptions['targetDims'])).astype(numpy.float32)
    for m in xrange(2000):
        saveTable(embeddingCoords,'unroll/embedding'+str(m).zfill(4)+'.csv')
        #step 1: get all distances
        t1 = time.time()
        msum = 0.
        for n in xrange(numchunks):
            distkernel(drv.In(rank_matrix[n]),
                       drv.In(embeddingCoords),
                       drv.Out(od[n]),
                       drv.Out(sums),
                       block=(threads,1,1))
            
            msum += sum(sums)
        #print time.time()-t1, " seconds to run DIST kernel."
        if m == 0:
            d0 = msum
            
        #do a STRESS2 check every so often to see if we should exit
        if m % 100:
            pass
        trg = d0/msum
        #step 2: Pool Adjacent Violators
        t2 = time.time()
        for n in xrange(numchunks):
            kernel(drv.In(od[n]),
                   drv.Out(d[n]),
                   numpy.float32(trg),
                   block=(threads,1,1))
        #print time.time()-t2, " seconds to run PAV kernel."
        t5 = time.time()
        for n in xrange(numchunks-1):
            if d[n][-1] > d[n+1][0]:
                imax = 1
                imin = len(d[n])-1
                runsum = d[n][-1]+d[n+1][0]
                runsize = 2.0
                changed = True
                while changed:
                    changed = False
                    while imax < len(d[n+1]) and runsum > d[n+1][imax]*runsize:
                        runsum += d[n+1][imax]
                        runsize += 1.0
                        imax += 1
                        changed = True
                    while imin > 0 and runsum < d[n][imin-1]*runsize:
                        imin -= 1
                        runsum += d[n][imin]
                        runsize += 1.0
                        changed = True
                
                d[n][imin:] = [runsum/runsize]*(len(d[n])-imin)
                d[n+1][:imax] = [runsum/runsize]*imax
        
        #scale everything ready to add to the points again
        for n in xrange(numchunks):
            scalekernel(drv.In(od[n]),
                       drv.In(d[n]),
                       drv.Out(d[n]),
                       block=(threads,1,1))
        #print time.time()-t5, " seconds to stitch and scale PAV sets."
        
        
        #step 3: modify positions
        t3 = time.time()
        for n in xrange(numchunks):
            deltakernel(drv.In(rank_matrix[n]),
                       drv.In(d[n]),
                       drv.In(embeddingCoords),
                       drv.Out(embeddingCoords),
                       block=(threads,1,1))
        #print time.time()-t3, " seconds to run Delta kernel."
        
        print "iter ",m," done"
    print time.time()-t0, " seconds to run NMDS."
    return embeddingCoords

def NMDSConfig(dataTable, targetDims, gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated APSP.
    """
    
    
    settings = dataConfig(dataTable,settings)
    
    settings["memSize"] = gpuMemSize*1024*1024
    settings["targetDims"] = targetDims
    
    #calculate memory constraints for the KNN chunks
    knnMem = (4*(settings["targetDims"]))*(settings["dataLength"]+4) #we have ( k * size( float + int ) * dataLength ) for our neighbour list
    chunkMem = settings["memSize"]-knnMem
    if chunkMem < settings["sourceDims"]*4*4:
        raise "GPU memory too small for KNN list!"
    chunkMem /= settings["sourceDims"]*settings["targetDims"]*4
    
    settings["chunkSize"] = min(chunkMem,settings["sourceDims"])
    
    settings["lastChunkSize"] = settings["dataLength"] % settings["chunkSize"]
    
    if settings["lastChunkSize"] > 0:
        settings["totalChunks"] = settings["dataLength"]/settings["chunkSize"]+1
    else:
        settings["totalChunks"] = settings["dataLength"]/settings["chunkSize"]
        settings["lastChunkSize"] = settings["chunkSize"]
    
    settings["block"] = (min(max(settings["dataLength"],1),512),1,1) #XXX: fix this for using the device data on max threads
    g1 = int(math.ceil(settings["dataLength"]/512.))
    g2 = int(math.ceil(g1/512.))
    g3 = int(math.ceil(g2/64.))
    settings["grid"] = (max(g1,1),max(g2,1),max(g3,1))
    return settings

def NMDSKernel(targetChunkSize,options,prefix = ''):
    """
    Return the string representation of the desired tiled KNN kernel.
    We do this twice for different target chunk sizes so we can handle odd length data.
    """
    chunkSize = targetChunkSize
    
    #a very naive NMDS implementation which may not work
    progstr = ("__global__ void NMDS(const unsigned int* RankChunk, const float* Coords, double* Adjustments, const long Coord_Offset) {\n"+
               "    const unsigned int v = threadIdx.x+blockIdx.x*512+blockIdx.y*512*512;\n" +
               "    if (v+Coord_Offset < "+str(options['sourceDims'])+" and v < "+str(options['chunkSize'])+") {\n" +
               "        double distance = 0.;\n" +
               "        const unsigned int vertex = (v+Coord_Offset)*"+str(options['targetDims'])+";\n" +
               "        const unsigned int vertex2 = (v)*"+str(options['targetDims'])+";\n" +
               "        for (unsigned int i = 0; i < "+str(options['targetDims'])+"; i++) {\n" +
               "            Adjustments[i+vertex2] = 0.;\n" +
               "        }\n"+
               "        for (unsigned int i = 1; i < "+str(options['sourceDims'])+"; i++) {\n" +
               "            const unsigned int p = (Rank[i+v*"+str(options['sourceDims'])+"])*"+str(options['targetDims'])+";\n" +
               "            double d = 0.0;\n" +
               "            for (unsigned int j = 0; j < "+str(options['targetDims'])+"; j++) {\n" +
               "                d += (Coords[j+vertex]-Coords[j+p])*(Coords[j+vertex]-Coords[j+p]);\n" +
               "            }\n" +
               "            d = sqrt(d);\n" +
               "            if (distance > d and vertex != p) {\n" +
               "                for (unsigned int j = 0; j < "+str(options['targetDims'])+"; j++) {\n" +
               "                    Adjustments[j+vertex] += (distance-d)*(Coords[j+vertex]-Coords[j+p])/d;\n" +
               "                }\n" +
               "            } else {\n" +
               "                distance = d;\n" +
               "            }\n" +
               "        }\n"+
               "    }\n" +
               "}\n")
    progstr =  ("__global__ void getDistChunks(const unsigned int* RankChunk, const unsigned float* distChunk, const float* Coords, float* Adjustments, const float distOffset) {\n"+
                "    const unsigned int v = threadIdx.x;\n"+
                "    if (v < "+str(chunkSize)+") {\n"+
                "        double origDist = 0.; //Calculate distance\n"+
                "        for (unsigned int j = 0; j < "+str(options['targetDims'])+"; ++j) {\n"+
                "            origDist += (Coords[RankChunk[2*v]*"+str(options['targetDims'])+"+j] - Coords[RankChunk[2*v+1]*"+str(options['targetDims'])+"+j])*\n"+
                "                        (Coords[RankChunk[2*v]*"+str(options['targetDims'])+"+j] - Coords[RankChunk[2*v+1]*"+str(options['targetDims'])+"+j]);\n"+
                "        }\n"+
                "        origDist = sqrt(origDist);\n"+
                "        distChunk[v] = max(origDist,distOffset); //Set initial distances\n"+
                "        __syncthreads();\n"+
                "        for (unsigned int i = 0; i < 100; ++i) { //XXX: 100 is a heuristic hack\n"+
                "            if (distChunk[v*2] > distChunk[v*2+1]) {\n"+
                "                distChunk[v*2+1] = distChunk[v*2];\n"+
                "            }\n"+
                "            __syncthreads();\n"+
                "        }\n"+
                "        distChunk[v] = distChunk[v]-origDist;\n"+
                "        __syncthreads();\n"+
                "        for (unsigned int i = 0; i < "+str(options['totalChunks'])+" and i*"+str(chunkSize)+"+v < "+str(options['sourceDims'])+"; ++i) {\n"+
                "            for (unsigned int j = 0; j < "+str(chunkSize)+"; ++j) {\n"+
                "                if (RankChunk[j*2] == i*"+str(chunkSize)+"+v and distChunk[j] > 0.) {\n"+
                "                    for (unsigned int k = 0; k < "+str(options['targetDims'])+"; ++k) {\n"+
                "                        Adjustments[RankChunk[2*j]*"+str(options['targetDims'])+"+k] += \n"+
                "                            (Coords[RankChunk[2*j]*"+str(options['targetDims'])+"+k] - \n"+
                "                             Coords[RankChunk[2*j+1]*"+str(options['targetDims'])+"+k])*\n"+
                "                             distChunk[j]*0.5;\n"+
                "                    }\n"+
                "                } else if (RankChunk[j*2+1] == i*"+str(chunkSize)+"+v and distChunk[j] > 0.) {\n"+
                "                    for (unsigned int k = 0; k < "+str(options['targetDims'])+"; ++k) {\n"+
                "                        Adjustments[RankChunk[2*j]*"+str(options['targetDims'])+"+k] -= \n"+
                "                            (Coords[RankChunk[2*j]*"+str(options['targetDims'])+"+k] - \n"+
                "                             Coords[RankChunk[2*j+1]*"+str(options['targetDims'])+"+k])*\n"+
                "                             distChunk[j]*0.5;\n"+
                "                    }\n"+
                "                }\n"+
                "            }\n"+
                "        }\n"+
                "    }\n"+
                "}\n")
    print progstr
    return progstr

def NMDS(datatable,nmdsOptions):
    rank_matrix = loadIntMatrix(datatable)
    
    t0 = time.time()
    nmds1 = SourceModule(NMDSKernel(nmdsOptions['chunkSize'],nmdsOptions))
    kernel = nmds1.get_function("NMDS")
    
    embeddingCoords = random.normal(0.,1.,(len(rank_matrix)*nmdsOptions['targetDims'])).astype(numpy.float64)
    adjustments = zeros(len(rank_matrix)*nmdsOptions['targetDims']).astype(numpy.float64)
    adj = zeros(nmdsOptions['chunkSize']*nmdsOptions['targetDims']).astype(numpy.float64)
    for i in xrange(2): #XXX: should calculate stress every reasonable number of steps
        for v in xrange(0,nmdsOptions['dataLength'],nmdsOptions['chunkSize']):
            kernel(drv.In(rank_matrix[v:(v+nmdsOptions['chunkSize'])].flatten().astype(numpy.uint32)),
                   drv.In(embeddingCoords),
                   drv.Out(adj),
                   drv.In(numpy.int64(v)),
                   grid=nmdsOptions['grid'],
                   block=nmdsOptions['block'])
            if v < nmdsOptions['totalChunks']:
                adjustments[v*nmdsOptions['targetDims']:(v+nmdsOptions['chunkSize'])*nmdsOptions['targetDims']] = adj
            else:
                adjustments[v*nmdsOptions['targetDims']:(v+nmdsOptions['chunkSize'])*nmdsOptions['targetDims']] = adj[:nmdsOptions['lastChunkSize']*nmdsOptions['targetDims']]
            print adjustments/len(adjustments)
        embeddingCoords += adjustments/len(adjustments) #*(2./len(adjustments))
    print time.time()-t0, " seconds to run NMDS."
    
    return embeddingCoords.reshape((nmdsOptions['sourceDims'],nmdsOptions['targetDims']))
