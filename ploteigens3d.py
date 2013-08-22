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
A simple plotting script for checking the swiss roll output data.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import sys

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
#ax = fig.add_subplot(111) #, projection='3d')
ax = fig.add_subplot(111, projection='3d')
n = 100
fname = 'embedding.csv'

if len(sys.argv) > 1:
    fname = sys.argv[1]
print fname
f = open(fname)
f = f.readlines()
x = []
y = []
z = []
col = []
for l in f:
    d = l.strip('\r\n').split(',')
    x.append(float(d[0]))
    y.append(float(d[1]))
    z.append(0.) #float(d[2]))
    #col.append(float(d[3]))
    
xmin = min(x)
xmax = max(x)
x = [(p-xmin)/(xmax-xmin+0.00000001) for p in x]
ymin = min(y)
ymax = max(y)
y = [(p-ymin)/(ymax-ymin+0.00000001) for p in y]
zmin = min(z)
zmax = max(z)
z = [(p-zmin)/(zmax-zmin+0.00000001) for p in z]

    
"""
for d in f[0].strip('\r\n').split(','):
    x.append(float(d))
for d in f[1].strip('\r\n').split(','):
    y.append(float(d))
for d in f[2].strip('\r\n').split(','):
    z.append(float(d))
"""
xs = np.array(x) #/2000.0
ys = np.array(y) #/2000.0
zs = np.array(z) #/2000.0


f = [(float(l.strip(' \r\n').split(',')[1])**2+float(l.strip(' \r\n').split(',')[0])**2)**0.5 for l in open('swissroll.csv').readlines()]
#f = range(len(x))
d = (max(f)-min(f))*1.0+0.00000000001
#print f


f# = [[0,1.-(i-min(f))/(d),(i-min(f))/(d)] for i in f]

f = [1.-(i-min(f))/(d)for i in f]

cs = np.array(f) #np.array([0.]*333+[0.25]*235+[0.5]*281+[0.75]*209) #range(len(x))) #np.array(col)
print len(cs),len(zs)
#ax.scatter(xs, ys,c=cs,s=20,cmap=matplotlib.cm.hot)

ax.scatter(xs, ys, zs, c=cs,s=20,cmap=matplotlib.cm.hsv) #cmap=matplotlib.cm.hot)
ax.view_init(70,-130)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

if len(sys.argv) > 1:
    plt.savefig(fname[:-3]+'png')
else:
    plt.show()
