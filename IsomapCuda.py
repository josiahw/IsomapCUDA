"""
Set of tools for doing Isomap

"""
import time
from numpy import array,zeros,amax,sqrt,dot
import numpy
from numpy.linalg import eig
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math

from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,saveTable


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
        for i in xrange(10):
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
    e = [(e[1].T[i]*sqrt(abs(e[0][i]))).tolist() for i in xrange(len(e[0]))]
    #e.reverse()
    e = [list(l) for l in zip(*e[:finalDims])]
    
    print time.time()-t0, " seconds to compute eigenvalue embedding."
    
    return e
    
