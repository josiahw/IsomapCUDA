#! /usr/bin/env python
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
A plotting script for generating the dimensional reconstruction error elbow.
"""
from DataUtils import loadMatrix
from numpy import array,corrcoef,dot,sqrt
import getopt,sys

arg_values = ['mindims=','maxdims=','if=', 'df=','of=','help','h']
optlist, args = getopt.getopt(sys.argv[1:], 'x', arg_values)

minDims = 1
maxDims = 20
infile='embedding.csv'
distfile='distances.csv'
outfile='embedding.ps'
saveFile=False


for o in optlist:
    if o[0].strip('-') == 'mindims':
        trgDims = int(o[1])
for o in optlist:
    if o[0].strip('-') == 'maxdims':
        srcDims = int(o[1])
for o in optlist:
    if o[0].strip('-') == 'df':
        distfile = o[1]
for o in optlist:
    if o[0].strip('-') == 'if':
        infile = o[1]
for o in optlist:
    if o[0].strip('-') == 'of':
        outfile = o[1]
        saveFile = True
for o in optlist:
    nonmetric = True
    
for o in optlist:
    if o[0].strip('-') == 'help' or o[1].strip('-') == 'h':
        print "The following commands are available:"
        print "\t--if=inputfile\tDefaults to embedding.csv"
        print "\t--of=outputfile\tDefaults to embedding.ps"
        print "\t--k=k_nearest_neighbours\tDefaults to 12"
        print "\t--outdims=embedding_dimensions\tDefaults to 3"
        print "\t--indims=input_dimensions\tDefaults to all in the input file"
        print "\t--nonmetric\tEnables non-metric MDS embeddings"

result = []

graph_distances = array(loadMatrix(distfile)).flatten()


for dim in xrange(minDims,maxDims):
    embedding = loadMatrix(infile)

    embedding_distances = []
    for i in xrange(len(embedding)):
        ei = embedding[i][:dim]
        for j in xrange(i):
            embedding_distances.append(embedding_distances[i+j*len(embedding)])
        for j in xrange(i,len(embedding)):
            e = ei-embedding[j][:dim]
            embedding_distances.append(dot(e,e))
    embedding_distances = corrcoef(sqrt(array(embedding_distances)),graph_distances)
    residual = (1 - embedding_distances*embedding_distances)[0][1]
    result.append(residual)
    print "residual: ",residual

