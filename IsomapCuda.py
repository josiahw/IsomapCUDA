"""
Set of tools for doing Isomap

"""
import time
from numpy import array,zeros,amax
import numpy
from numpy.linalg import eig
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,saveTable


# KNN Algorithm --------------------------------------------

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
    
    program = ("__global__ void "+prefix+"knn(const float* values, const float* test_against, float* distances, unsigned int* indices, const long offset, const long offset2) {\n" +
               "    const unsigned int v = threadIdx.x;\n" +
               "    const unsigned int dist_nn = (v+offset2)*"+str(options['k'])+";\n" +
               "    const unsigned int dist_ds = v*"+str(options['sourceDims'])+";\n" +
               "    for (unsigned int i = 0; i < "+str(targetChunkSize)+"; i++) {\n" +
               "        double distance = 0.;\n" + 
               "        for (unsigned int j = 0; j < "+str(options['sourceDims'])+"; j++) {\n" +
               "            distance += (test_against[i*"+str(options['sourceDims'])+"+j]-values[dist_ds+j])*(test_against[i*"+str(options['sourceDims'])+"+j]-values[dist_ds+j]);\n" +
               "        }\n" +
               "        distance = sqrt(distance);\n" +
               "        if (distance <= distances[dist_nn+"+str(options['k'])+"-1] and distance < "+str(options['eps'])+" and i+offset != v+offset2) {\n" +
               "            unsigned int j = 0;\n" +
               "            while (distance > distances[dist_nn+j]) {\n" +
               "                j++;\n" +
               "            }\n" +
               "            for (int k = "+str(options['k'])+"-2; k >= j; k--) {\n" +
               "                distances[dist_nn+k+1] = distances[dist_nn+k];\n" +
               "                indices[dist_nn+k+1] = indices[dist_nn+k];\n" +
               "            }\n" +
               "            distances[dist_nn+j] = distance;\n" +
               "            indices[dist_nn+j] = i+offset;\n" +
               "        }\n" +
               "    }\n" +
               "}\n")
    print program
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
            prg(drv.In(source),drv.In(data[t]),drv.InOut(dists[t]),drv.InOut(indices[t]),numpy.int64(t*knnOptions['chunkSize']),numpy.int64(offset2*knnOptions['chunkSize']),block=(knnOptions['chunkSize'],1,1))
            offset2 += 1
        prg(drv.In(data[-1]),drv.In(data[t]),drv.InOut(dists[t]),drv.InOut(indices[t]),numpy.int64(t*knnOptions['chunkSize']),numpy.int64(offset2*knnOptions['chunkSize']),block=(knnOptions['lastChunkSize'],1,1))
    t = len(data)-1
    offset2 = 0
    for source in data[:-1]:
        prg2(drv.In(source),drv.In(data[t]),drv.InOut(dists[t]),drv.InOut(indices[t]),numpy.int64(t*knnOptions['chunkSize']),numpy.int64(offset2*knnOptions['chunkSize']),block=(knnOptions['chunkSize'],1,1))
        offset2 += 1
    prg2(drv.In(data[-1]),drv.In(data[t]),drv.InOut(dists[t]),drv.InOut(indices[t]),numpy.int64(t*knnOptions['chunkSize']),numpy.int64(offset2*knnOptions['chunkSize']),block=(knnOptions['lastChunkSize'],1,1))
    
    #organise data and add neighbours
    alldists = []
    allindices = []
    for i in xrange(knnOptions['totalChunks']): #rearrange into single lists
        alldists.extend(dists[t].reshape((knnOptions['chunkSize'],knnOptions['k'])))
        allindices.extend(indices[t].reshape((knnOptions['chunkSize'],knnOptions['k'])))
    alldists = alldists[:-(knnOptions['chunkSize']-knnOptions['lastChunkSize'])] #.tolist()
    allindices = allindices[:-(knnOptions['chunkSize']-knnOptions['lastChunkSize'])] #.tolist()
    print allindices[0]
    for i in xrange(len(alldists)): #remove excess entries
        if knnOptions['eps'] in alldists[i]:
            ind = alldists[i].tolist().index(knnOptions['eps'])
            alldists[i] = alldists[i][:ind].tolist()
            allindices[i] = allindices[i][:ind].tolist()
            j = 0
            for ind in allindices[i]: #add mirrored entries
                if not (i in allindices[ind]):
                    allindices[ind].append(i)
                    alldists[ind].append(alldists[i][j])
                j += 1
    
    maxKValues = max([len(p) for p in allindices]) 
    
    for i in xrange(len(alldists)): #pad all entries to the same length for the next algorithm
        if len(alldists[i]) < maxKValues:
            alldists[i].extend( [knnOptions['eps']]*(maxKValues-len(alldists[i])) )
            allindices[i].extend( [0]*(maxKValues-len(alldists[i])) )
    
    print time.time()-t0, " seconds to process KNN"
    print allindices[-2]
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
    
    settings["block"] = (max(settings["dataLength"]/512,1),max(settings["dataLength"]/512/512,1),max(settings["dataLength"]/512/512/64,1))
    return settings
    

def APSPKernel(targetChunkSize,options,prefix = ''):
    """
    Return the string representation of the desired tiled KNN kernel.
    We do this twice for different target chunk sizes so we can handle odd length data.
    """
    
    #implemented from paper
    progstr = ("__global__ void SSSP(const unsigned int* Edges, const float* Weights, const float* Costs, float* Paths) {\n"+
               "    const unsigned int v = threadIdx.x+threadIdx.y*512+threadIdx.z*512*64;\n" +
               "    //for (unsigned int j = 0; j < "+str(targetChunkSize)+"; j++) {\n" +
               "    if (v < "+str(options['dataLength'])+") {\n" +
               "        const unsigned int vertex = v; //+j*"+str(options['dataLength'])+";\n" +
               "        float p = Costs[vertex];\n" +
               "        for (unsigned int i = 0; i < "+str(options['k']-(options['k']%2))+"; i += 2) {\n" +
               "            const unsigned int nid = vertex*"+str(options['k'])+"+i;\n" +
               "            p = min(p,min(Costs[Edges[nid]]+Weights[nid],Costs[Edges[nid+1]]+Weights[nid+1]));\n" +
               "        }\n")
               
    if options['k']%2: #minor parallel speedup by checking 2 at once
        progstr +=("        p = min(p,Costs[Edges[vertex*"+str(options['k'])+"+"+str(options['k']-1)+"]]+Weights[vertex*"+str(options['k'])+"+"+str(options['k']-1)+"]);\n")
    
    progstr +=("        Paths[vertex] = p;\n" +
               "    }\n" +
               "    //}\n" +
               "}\n")
    print progstr
    return progstr


def APSP(dataTable,apspOptions):
    knn_refs,knn_dists = loadSplitTable(dataTable,apspOptions)
    
    #neighbours = numpy.int32(self.number_of_neighbours)
    
    sssp1 = SourceModule(APSPKernel(apspOptions['chunkSize'],apspOptions))
    kernel = sssp1.get_function("SSSP")
    
    #print str(self.data_length/self.num_threads),self.data_length/self.num_threads*200
    
    Costs0 = array([apspOptions['eps']]*apspOptions['dataLength']).astype(numpy.float32)
    Matrix = []
    t0 = time.time()
    
    #iterate through every row of the path cost matrix
    last = 130
    for v in xrange(0,apspOptions['dataLength'],apspOptions['chunkSize']):
        
        #create a new row for the cost matrix
        Costs = Costs0.copy()
        Costs[v] = 0.
        for n in xrange(v):
            Costs[n] = Matrix[n][v]
        v2 = v*apspOptions['k']
        
        #initialise the costs we have for the immediate neighbours
        for n in xrange(apspOptions['k']):
            print v2+n,knn_refs[v2+n]
            Costs[knn_refs[v2+n]] = knn_dists[v2+n]
        
        #iteratively expand the shortest paths (1 iter per kernel run) until we have all the paths
        for i in xrange(last-5):
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs),drv.InOut(Costs),block=apspOptions['block'])
        l = last-5
        
        #XXX: this is expensive, find a better way
        while amax(Costs) > 100000.:
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs),drv.InOut(Costs),block=apspOptions['block'])
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs),drv.InOut(Costs),block=apspOptions['block'])
            l += 2
        last = l
        
        #do a few extra iterations in case we missed the shortest paths
        for i in xrange(40):
            kernel(drv.In(knn_refs),drv.In(knn_dists),drv.In(Costs),drv.InOut(Costs),block=apspOptions['block'])
        
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
    settings["eps"] = eps
    
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
    
    settings["block"] = (max(settings["dataLength"]/512,1),max(settings["dataLength"]/512/512,1),max(settings["dataLength"]/512/512/64,1))
    return settings

def NormMatrix(dataTable,nmOptions):
        t0 = time.time()
        
        normedMatrix = loadMatrix(dataTable)
        
        progstr = ("__global__ void SumSquare(const unsigned int totalNodes, float* Sums, float* Paths) {\n"+
                   "    const unsigned int v = threadIdx.x+threadIdx.y*512+threadIdx.z*64;\n" +
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
            prg(drv.In(numpy.uint32(nmOptions['dataLength'])),drv.InOut(normsums),drv.InOut(normsample),block=nmOptions['block'])
            normedMatrix[i*nmOptions['chunkSize']:(i+1)*nmOptions['chunkSize']] = normsample.reshape((nmOptions['chunkSize'],nmOptions['dataLength']))
        
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
        
    e = eigh(loadMatrix(dataTable))
    e = [(e[1].T[i]*sqrt(abs(e[0][i]))).toList() for i in xrange(finalDims)]
    e.reverse()
    
    print time.time()-t0, " seconds to compute eigenvalue embedding."
    
    return e
    
