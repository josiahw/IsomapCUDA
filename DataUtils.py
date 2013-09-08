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
Data Load/Save/Formatting utilities for use with the GPU isomap algorithms.
"""
from numpy import array,savetxt
import numpy

def dataConfig(dataTable,settings = {}):
    """
    Creates all the data settings for higher algorithms.
    """
    if type( "" ) == type( dataTable ):
        fi = open(dataTable)
        l = fi.readline().strip('\n ').split(',')
        fi.close()
        settings["sourceDims"] = len(l)
        fi = open(dataTable)
        f = fi.readlines()
        fi.close()
        if len(f[-1]) > 1:
            f = len(f)
        else:
            f = len(f)-1
        settings["dataLength"] = f
    else:
        settings["sourceDims"] = len(dataTable)
        if len(dataTable.shape) > 1:
            settings["sourceDims"] = len(dataTable[0])
        settings["dataLength"] = len(dataTable)
    return settings

def loadTable(dataTable,options):
    """
    Load and chunkify a table either as a list/numpy matrix or as a csv file
    """
    results = []
    if type( "" ) == type( dataTable ):
        f = open(dataTable)
        l = f.readlines()
        f.close()
        while len(l) > options["chunkSize"]:
            chunk = []
            for v in l[:options["chunkSize"]]:
                chunk.append(array([float(p) for p in v.strip('\r\n ').split(',')[:options["sourceDims"]]])) #chopping by sourceDims lets us ignore extra columns if we want
            l = l[options["chunkSize"]:]
            results.append(array(chunk).astype(numpy.float32))
        chunk = []
        for v in l:
            chunk.append(array([float(p) for p in v.strip('\r\n ').split(',')[:options["sourceDims"]]]))
        results.append(array(chunk).astype(numpy.float32))
    else:
        dt = array(dataTable).astype(numpy.float32)
        for i in xrange(len(dt)/options["chunkSize"]):
            results.append(dt[i*options["chunkSize"]:(i+1)*options["chunkSize"],:options["sourceDims"]].astype(numpy.float32))
        results.append(dt[-(len(dt)%options["chunkSize"]):,:options["sourceDims"]].astype(numpy.float32))
    return results

def loadSplitTable(dataTable):
    """
    Load a table either as a list/numpy matrix or as a csv file
    """
    results1 = []
    results2 = []
    if type( "" ) == type( dataTable ):
        f = open(dataTable)
        l = f.readlines()
        f.close()
        for v in l:
            l = v.strip('\r\n ').split(',')
            results1.append(array([int(p) for p in l[:len(l)/2]]).astype(numpy.uint32))
            results2.append(array([float(p) for p in l[len(l)/2:]]).astype(numpy.float32))
    else:
        results1 = array(dataTable)[:,:len(dataTable[0])/2].astype(numpy.uint32)
        results2 = array(dataTable)[:,len(dataTable[0])/2:].astype(numpy.float32)
    return [results1,results2]

def loadMatrix(dataTable):
    """
    Load a table either as a list/numpy matrix or as a csv file
    """
    results = []
    if type( "" ) == type( dataTable ):
        results = numpy.loadtxt(dataTable,delimiter=',').astype(numpy.float32)
        #f = open(dataTable)
        #l = f.readlines()
        #f.close()
        #for v in l:
        #    results.append(array([float(p) for p in v.strip('\r\n ').split(',')]).flatten().astype(numpy.float32))
    else:
        results = array(dataTable).astype(numpy.float32)
    return results

def loadIntMatrix(dataTable):
    """
    Load a table either as a list/numpy matrix or as a csv file
    """
    results = []
    if type( "" ) == type( dataTable ):
        results = numpy.loadtxt(dataTable,delimiter=',').astype(numpy.uint32)
        #f = open(dataTable)
        #l = f.readlines()
        #f.close()
        #for v in l:
        #    results.append(array([float(p) for p in v.strip('\r\n ').split(',')]).flatten().astype(numpy.uint32))
    else:
        results = array(dataTable).astype(numpy.uint32)
    return results

def saveTable(dataTable,filename='data.csv'):
    savetxt(filename, array(dataTable), delimiter=',')
