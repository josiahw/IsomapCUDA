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
GPU based K Means algorithm.
"""

import time
from numpy import array,zeros,amax,amin,sqrt,dot
import numpy
import random
from numpy.linalg import eig
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math


from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,loadIntMatrix,saveTable

KernelLocation = "CudaKernels/KMEANS/"

# KMeans Algorithm --------------------------------------------

def KMeansConfig(dataTable, k, eps = 0.00001, srcDims=100000000000,gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated KMeans.
    """
    settings = dataConfig(dataTable,settings)
    settings["sourceDims"] = min(settings["sourceDims"],srcDims)
    
    #XXX: determine memory and thread sizes from device
    settings["memSize"] = gpuMemSize*1024*1024
    settings["maxThreads"] = 1024
    
    #set up chunk sizes
    memoryPerElement = 4*(settings["sourceDims"]*2+2) + 20*4 #this is an estimated memory used per element
    settings["chunkSize"] = min(int(math.ceil(float(settings["memSize"])/memoryPerElement)),settings["dataLength"])
    settings["lastChunkSize"] = ((settings["dataLength"]-1) % settings["chunkSize"]) + 1
    
    #create kernel gridsize tuples
    settings["block"] = (settings["maxThreads"],1,1)
    settings["grid"] = (max(int(math.ceil(float(settings["chunkSize"])/settings["maxThreads"])),1),1,1)
    
    #precalculate all constant kernel params
    settings["dimensions"] = numpy.int64(settings["sourceDims"])
    settings["k"] = numpy.int64(k)
    settings["eps"] = numpy.float32(eps)
    settings["dataSize"] = numpy.int64(settings["dataLength"])
    settings["chunkSize"] = numpy.int64(settings["chunkSize"])
    settings["maxThreads"] = numpy.int64(settings["maxThreads"])
    
    return settings

def KMeans(dataTable, k, epsilon=0.00001, srcDims = 1000000000000000, iters=20, normData = False):
    """
    Get the best out of iters tries of k means terminating when delta k < epsilon
    """
    #load up the configuration
    kmOptions = KMeansConfig(dataTable,k,epsilon,srcDims)
    
    
    #load and format the table for use.
    data = loadMatrix(dataTable)[:,:kmOptions['sourceDims']]
    
    #check if we should normalise the data (this is really quick and dirty, replace it with something better)
    if normData:
        dmax = amax(data)
        dmin = amin(data)
        data = (data-dmin)/(dmax-dmin+0.00000001)
    
    
    #make our starting point solutions from the dataset
    solutions = [array(random.sample(data,k)) for i in xrange(iters)]
    
    #chunk solutions if necessary
    for i in xrange(len(solutions)):
        sol = []
        while len(solutions[i]) > kmOptions['chunkSize']:
            sol.append(solutions[i][:kmOptions['chunkSize']])
            solutions[i] = solutions[i][kmOptions['chunkSize']:]
        sol.append(solutions[i])
        solutions[i] = sol
    
    #create our chunked problem data
    dataChunks = []
    while len(data) > kmOptions['chunkSize']:
        dataChunks.append(data[:kmOptions['chunkSize']])
        data = data[kmOptions['chunkSize']:]
    dataChunks.append(data)
    kNorm = (len(dataChunks)-1)+len(dataChunks[-1])/float(len(dataChunks[0]))
    
    #create the CUDA kernels
    program = SourceModule(open(KernelLocation+"KMEANS_LABEL.nvcc").read())
    prg = program.get_function("KMEANS_LABEL")
    program = SourceModule(open(KernelLocation+"KMEANS_UPDATE.nvcc").read())
    prg2 = program.get_function("KMEANS_UPDATE")
    t0 = time.time()
    
    #store the resultant performance of each solution here
    results = []
    finalSols = []
    
    #make GPU allocations and support variables
    total = 0.
    dists = [numpy.zeros(kmOptions['chunkSize']).astype(numpy.float32)+10000000000000000. for i in xrange(len(dataChunks))] #this is used as an intermediate step
    labels = [numpy.zeros(kmOptions['chunkSize']).astype(numpy.uint32) for i in xrange(len(dataChunks))] #this is used as an intermediate step
    data_gpu = drv.mem_alloc(dataChunks[0].nbytes)
    k_gpu = drv.mem_alloc(solutions[0][0].nbytes)
    labels_gpu = drv.mem_alloc(labels[0].nbytes)
    dists_gpu = drv.mem_alloc(dists[0].nbytes)
    
    #calculate KMeans
    for sol in solutions:
        t0 = time.time()
        for i in xrange(10000):
            #Step 1: find all the closest labels
            for i in xrange(len(sol)):
                #copy in blank distances, labels, and the label coordinates
                drv.memcpy_htod(k_gpu, sol[i])
                for j in xrange(len(dataChunks)):
                    drv.memcpy_htod(data_gpu, dataChunks[j])
                    drv.memcpy_htod(labels_gpu, labels[j])
                    drv.memcpy_htod(dists_gpu, dists[j])
                    prg(k_gpu,
                        data_gpu,
                        kmOptions["dimensions"],
                        labels_gpu,
                        dists_gpu,
                        kmOptions['k'],
                        kmOptions['dataSize'],
                        kmOptions['chunkSize'],
                        numpy.int64(i*kmOptions['chunkSize']), #k offset
                        numpy.int64(j*kmOptions['chunkSize']), #data offset
                        kmOptions['maxThreads'],
                        block=kmOptions['block'],
                        grid=kmOptions['grid'])
                drv.memcpy_dtoh(labels[i], labels_gpu)
            #Step 2: find the new averages
            old_sol = [s.copy() for s in sol]
            for i in xrange(len(sol)):
                #load up a blank set of k matrices
                drv.memcpy_htod(k_gpu, sol[i]*0.)
                for j in xrange(len(dataChunks)):
                    drv.memcpy_htod(data_gpu, dataChunks[j])
                    drv.memcpy_htod(labels_gpu, labels[j])
                    prg2(k_gpu,
                        data_gpu,
                        kmOptions["dimensions"],
                        labels_gpu,
                        kmOptions['k'],
                        kmOptions['dataSize'],
                        kmOptions['chunkSize'],
                        numpy.int64(i*kmOptions['chunkSize']), #label offset
                        numpy.int64(j*kmOptions['chunkSize']), #data offset
                        kmOptions['maxThreads'],
                        block=kmOptions['block'],
                        grid=kmOptions['grid'])
                drv.memcpy_dtoh(sol[i], k_gpu)
                sol[i] /= kNorm #final normalisation
            #Step 3: check that the update distance is larger than epsilon
            total = 0.
            for j in xrange(len(sol)):
                tmp = sol[j]-old_sol[j]
                tmp = tmp*tmp
                total += sum([sum(t**0.5) for t in tmp])
            if total/kmOptions['dataSize'] < kmOptions['eps']:
                break
        print "solution done in ",time.time()-t0
        results.append((total,len(results)))
        finalSols.append(numpy.concatenate(sol)[:kmOptions['dataSize']])
    results.sort()
    return finalSols[results[0][1]]



if __name__ == '__main__':
    import getopt,sys
    arg_values = ['dims=','if=','of=','k=','eps=','help','h']
    optlist, args = getopt.getopt(sys.argv[1:], 'x', arg_values)
    
    srcDims = 10000000000000
    k = -1
    eps = 0.00001
    infile='swissroll.csv'
    outfile='swissmeans.csv'
    
    for o in optlist:
        if o[0].strip('-') == 'dims':
            srcDims = int(o[1])
    for o in optlist:
        if o[0].strip('-') == 'if':
            infile = o[1]
    for o in optlist:
        if o[0].strip('-') == 'of':
            outfile = o[1]
    for o in optlist:
        if o[0].strip('-') == 'k':
            k = int(o[1])
    for o in optlist:
        if o[0].strip('-') == 'eps':
            eps = float(o[1])
    
    if k < 0:
        k = len(open(infile).readlines())/10
    for o in optlist:
        if o[0].strip('-') == 'help' or o[1].strip('-') == 'h':
            print "The following commands are available:"
            print "\t--if=inputfile\tDefaults to swissroll.csv"
            print "\t--of=outputfile\tDefaults to swissmeans.csv"
            print "\t--k=k_nearest_neighbours\tDefaults to dataset_length/10"
            print "\t--dims=input_dimensions\tDefaults to all in the input file"
            print "\t--eps=termination_epsilon\tDefaults to 0.00001"
    result = KMeans(infile,k,eps,srcDims)
    
    
    saveTable(result,outfile)

