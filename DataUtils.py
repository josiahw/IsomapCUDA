"""
Data Utils for use with isomap.
"""
from numpy import array
import numpy

def dataConfig(dataTable,settings = {}):
    """
    Creates all the data settings for higher algorithms.
    """
    if type( "" ) == type( dataTable ):
        fi = open(dataTable)
        l = [float(p) for p in fi.readline().strip('\n ').split(',')]
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
                chunk.append(array([float(p) for p in v.strip('\r\n ').split(',')[:options["sourceDims"]]]).flatten()) #chopping by sourceDims lets us ignore extra columns if we want
            l = l[options["chunkSize"]:]
            results.append(array(chunk))
        for v in l:
            chunk.append(array([float(p) for p in v.strip('\r\n ').split(',')[:options["sourceDims"]]]).flatten())
        results.append(array(chunk).astype(numpy.float32))
    else:
        dt = array(dataTable).astype(numpy.float32)
        for i in xrange(len(dt)/options["chunkSize"]):
            results.append(dt[i*options["chunkSize"]:(i+1)*options["chunkSize"],:options["sourceDims"]].flatten())
        results.append(dt[-(len(dt)%options["chunkSize"]):,:options["sourceDims"]].flatten())
    return results

def loadSplitTable(dataTable,options):
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
            results1.append(array([int(p) for p in l[:len(l)/2]]).flatten().astype(numpy.uint32))
            results2.append(array([float(p) for p in l[len(l)/2:]]).flatten().astype(numpy.float32))
    else:
        results1 = array(dataTable)[:,:len(dataTable[0])/2].flatten().astype(numpy.uint32)
        results2 = array(dataTable)[:,len(dataTable[0])/2:].flatten().astype(numpy.float32)
    return [results1,results2]

def loadMatrix():
    """
    Load a table either as a list/numpy matrix or as a csv file
    """
    results = []
    if type( "" ) == type( dataTable ):
        f = open(dataTable)
        l = f.readlines()
        f.close()
        for v in l:
            results.append(array([float(p) for p in v.strip('\r\n ').split(',')]).flatten().astype(numpy.float32))
    else:
        results = array(dataTable).astype(numpy.float32)
    return results

def saveTable(dataTable,filename='data.csv'):
    f = open(filename,'w')
    
    for d in dataTable:
        f.write(str(list(d)).strip('[] ') + '\n')
    f.close()
