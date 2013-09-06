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
GPU based Non-metric MDS (using the Kruskal algorithm)
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

KernelLocation = "CudaKernels/NMDS/"

def NMDSConfig(dataTable, targetDims, gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated APSP.
    """
    #XXX: this needs replacing
    
    settings = dataConfig(dataTable,settings)
    
    settings["memSize"] = gpuMemSize*1024*1024
    settings["targetDims"] = targetDims
    
    
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
    
    settings["block"] = (min(max(settings["dataLength"],1),1024),1,1) #XXX: fix this for using the device data on max threads
    g1 = int(math.ceil(settings["dataLength"]/512.))
    g2 = int(math.ceil(g1/512.))
    g3 = int(math.ceil(g2/64.))
    settings["grid"] = (max(g1,1),max(g2,1),max(g3,1))
    return settings

#Non-metric MDS algorithm -------------------------------------------------------
def NMDS(dataTable,origData, origDims, trgDims):
    rank_matrix = dataTable
    
    nmdsOptions = NMDSConfig(dataTable,trgDims)
    nmdsOptions['sourceDims'] = len(origData)
    
    numchunks = len(rank_matrix)/150000 + (len(rank_matrix)%150000>0) #len(rank_matrix)/200000
    chunksize = int(math.ceil(len(rank_matrix)/float(numchunks)))
    
    #if active, we will use the origdist matrix for adjustments
    usemetric = False
    
    #set alpha
    alpha = 0.35
    """
    #precalculate the stress denominator
    av = 0.0
    for i in xrange(1,len(origData)):
        for j in xrange(i,len(origData)):
            av += (sqrt(dot((origData[i]-origData[j]),(origData[i]-origData[j]))))
    av /= len(rank_matrix)
    stressdenominator = 0.
    for i in xrange(1,len(origData)):
        for j in xrange(i,len(origData)):
            stressdenominator += (sqrt(dot((origData[i]-origData[j]),(origData[i]-origData[j])))-av)**2
    """
    """
    #create a start point lookup for accelerating NMDS
    startpts = []
    ctr = 0
    for i in xrange(1,len(origData)):
        startpts.append(ctr)
        ctr += len(origData)-i
    startpts = array(startpts).astype(numpy.uint32)
    """
    #split rank matrix
    rm = [rank_matrix[i*chunksize:(i+1)*chunksize].astype(numpy.uint32) for i in xrange(numchunks)]
    del rank_matrix
    rank_matrix = rm
    t0 = time.time()
    
    reordered_pts = []
    pt_starts = []
    for chunk in rank_matrix:
        reorder_count = [0]*2*nmdsOptions['sourceDims']
        reorder = [[] for i in xrange(2*nmdsOptions['sourceDims'])]
        for i in xrange(0,len(chunk),1):
            reorder_count[chunk[i][0]] += 1
            reorder[chunk[i][0]].append(i/2)
            reorder_count[chunk[i][1]+nmdsOptions['sourceDims']] += 1
            reorder[chunk[i][1]+nmdsOptions['sourceDims']].append(i/2)
        counter = 0
        final = []
        for ct in reorder_count:
            final.append(counter)
            counter += ct
        final.append(counter)
        pt_starts.append(array(final).astype(numpy.uint32))
        final = []
        for ct in reorder:
            final.extend(ct)
        reordered_pts.append(array(final).astype(numpy.uint32))
    
    #prepare an embedding and adjustments
    embeddingCoords = array(origData)[:,:nmdsOptions['targetDims']].astype(numpy.float32) + random.normal(0.,.01,(nmdsOptions['sourceDims'],nmdsOptions['targetDims'])).astype(numpy.float32)
    #del origData
    threads = 1024
    
    #XXX: make these go away sometime, make kernels configurable on the fly
    KernelHeader = ('#define DATA_SIZE ('+str(nmdsOptions['sourceDims'])+')\n'+
                    '#define TOTAL_THREADS ('+str(threads)+')\n'+
                    '#define DATA_STEP_SIZE ('+str(int(math.ceil(nmdsOptions['sourceDims']/float(threads))))+')\n'+
                    '#define DATA_LENGTH ('+str(nmdsOptions['sourceDims'])+')\n'+
                    '#define DATA_DIMS ('+str(nmdsOptions['targetDims'])+')\n'+
                    '#define ALPHA (0.25)\n')
    LastKernelHeader = ('#define CHUNK_SIZE ('+str(len(rank_matrix[-1]))+')\n'+
                             '#define STEP_SIZE ('+str(int(math.ceil(len(rank_matrix[-1])/float(threads))))+')\n'+
                             KernelHeader)
    KernelHeader = ('#define CHUNK_SIZE ('+str(chunksize)+')\n'+
                    '#define STEP_SIZE ('+str(int(math.ceil(chunksize/float(threads))))+')\n'+
                    KernelHeader)
    
    PAVkernel = KernelHeader + open(KernelLocation+"PAV.nvcc").read()
    DISTkernel = KernelHeader + open(KernelLocation+"RANKDIST.nvcc").read()
    NMDSkernel = open(KernelLocation+"NMDS.nvcc").read()
    SCALEkernel = open(KernelLocation+"SCALE.nvcc").read()
    lastPAVkernel = LastKernelHeader + open(KernelLocation+"PAV.nvcc").read()
    pk = SourceModule(PAVkernel)
    dk = SourceModule(DISTkernel)
    nk = SourceModule(NMDSkernel)
    #sk = SourceModule(SCALEkernel)
    kernel = pk.get_function("PAV")
    distkernel = dk.get_function("RankDist")
    nmdskernel = nk.get_function("NMDS")
    #scalekernel = sk.get_function("Scale")
    lpk = SourceModule(lastPAVkernel)
    lkernel = lpk.get_function("PAV")
    deltasums = 10.
    sm = 100000000000000.
    d = [zeros(chunksize).astype(numpy.float32) for n in xrange(numchunks)]
    od = [zeros(chunksize).astype(numpy.float32) for n in xrange(numchunks)]
    sums = zeros(threads).astype(numpy.float32)
    deltas = zeros((chunksize,nmdsOptions['targetDims'])).astype(numpy.float32)
    
    #gpu data stores
    sums_gpu = drv.mem_alloc(sums.nbytes)
    #deltas_gpu = drv.mem_alloc(deltas.nbytes)
    dists_gpu = drv.mem_alloc(d[0].nbytes)
    #odists_gpu = drv.mem_alloc(od[0].nbytes)
    reorderPts_gpu = drv.mem_alloc(reordered_pts[0].nbytes)
    reorderStarts_gpu = drv.mem_alloc(pt_starts[0].nbytes)
    ranks_gpu = drv.mem_alloc(rank_matrix[0].nbytes)
    embedding_gpu = drv.mem_alloc(max(embeddingCoords.nbytes,rank_matrix[0].nbytes))
    
    #counter
    m = 0
    
    #blocksize for chunks
    blk = (1024,chunksize/1024+(chunksize%1024>0),1)
    elementBlk = (1024,2*nmdsOptions['sourceDims']/1024+(2*nmdsOptions['sourceDims']%1024>0),1) #nmdsOptions['targetDims'])
    drv.memcpy_htod(embedding_gpu, embeddingCoords)
    while deltasums > 1.007 and m < 2000:
        #saveTable(embeddingCoords,'unroll/embedding'+str(m).zfill(4)+'.csv')
        #step 1: get all distances
        t1 = time.time()
        msum = 0.
        
        """
        #do a STRESS2 check every so often to see if we should exit
        if not m % 100 and m > 700:
            old_sm = sm
            p = q = 0
            sm = 0.
            for i in xrange(1,len(origData)):
                for j in xrange(i,len(origData)):
                    if q >= len(od[p]):
                        q = q % len(od[p])
                        p += 1
                    sm += (od[p][q]-sqrt(dot((origData[i]-origData[j]),(origData[i]-origData[j]))))**2
                    q += 1
            sm /= stressdenominator
            print "STRESS2: ", sm
            if sm > old_sm*1.1:
                break
                #usemetric = True
            sm = min(sm,old_sm)
        """
        sums *= 0.;
        drv.memcpy_htod(sums_gpu, sums)
        for n in xrange(numchunks):
            drv.memcpy_htod(ranks_gpu, rank_matrix[n])
            distkernel(ranks_gpu,
                       embedding_gpu,
                       dists_gpu,
                       sums_gpu,
                       numpy.int64(len(d[n])),
                       numpy.int64(nmdsOptions['targetDims']),
                       block=(threads,1,1))
            drv.memcpy_dtoh(od[n], dists_gpu)
        drv.memcpy_dtoh(sums, sums_gpu)
        msum = sum(sums)
        #print time.time()-t1, " seconds to run DIST kernel."
        if m == 0:
            d0 = msum
        trg = d0/msum
        
        if usemetric: #this does iterative metric NMDS
            for i in xrange(numchunks):
                d[i] = origDistance[i*chunksize:(i+1)*chunksize]
        else: #else do normal monotonic NMDS
            
            #step 2: Pool Adjacent Violators
            t2 = time.time()
            for n in xrange(numchunks-1):
                drv.memcpy_htod(dists_gpu, od[n])
                kernel(dists_gpu,
                       block=(threads,1,1))
                drv.memcpy_dtoh(d[n], dists_gpu)
            drv.memcpy_htod(dists_gpu, od[-1])
            lkernel(dists_gpu,
                    block=(threads,1,1))
            drv.memcpy_dtoh(d[-1], dists_gpu)
            
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
        scaleFactor = numpy.float32(alpha/(nmdsOptions['sourceDims']-1.0))
        for n in xrange(numchunks):
            d[n] /= od[n] #(scaleFactor*(1.-d[n]/od[n]))
            """
            drv.memcpy_htod(ranks_gpu, od[n])
            drv.memcpy_htod(dists_gpu, d[n])
            scalekernel(ranks_gpu,
                        dists_gpu,
                        numpy.int64(len(d[n])),
                        scaleFactor,
                        block=(1024,1,1),grid=blk)
            drv.memcpy_dtoh(d[n],dists_gpu)
            """
        #print time.time()-t5, " seconds to stitch and scale PAV sets."
        
        #step 3: modify positions
        t3 = time.time()
        
        
        for n in xrange(numchunks):
            drv.memcpy_htod(reorderPts_gpu, reordered_pts[n])
            drv.memcpy_htod(reorderStarts_gpu, pt_starts[n])
            drv.memcpy_htod(ranks_gpu, rank_matrix[n])
            drv.memcpy_htod(dists_gpu, d[n])
            nmdskernel( ranks_gpu, dists_gpu,
                        embedding_gpu,
                        reorderPts_gpu,reorderStarts_gpu,
                        numpy.int64(len(d[n])),
                        numpy.int64(nmdsOptions['targetDims']),
                        numpy.int64(nmdsOptions['sourceDims']),
                        scaleFactor, numpy.float32(trg),
                        block=(1024,1,1),grid=elementBlk)
        drv.memcpy_dtoh(embeddingCoords,embedding_gpu)
        #print time.time()-t3, " seconds to run Delta kernel."
        
        m += 1
        #print time.time()-t1, " seconds to run NMDS iter."
        if m % 20 == 0:
            deltasums = (deltasums+sum([sum(point*point) for point in d])/(1250*2497))/2.
            print deltasums, "\t(Target is < 1.007)"
        
        #print "iter ",m," done"
    print time.time()-t0, " seconds to run NMDS."
    print m
    return embeddingCoords


