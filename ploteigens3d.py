import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
#ax = fig.add_subplot(111) #, projection='3d')
ax = fig.add_subplot(111, projection='3d')
n = 100

f = open('embedding.csv')

f = f.readlines()
x = []
y = []
z = []
col = []
print len(f)
for l in f:
    d = l.strip('\r\n').split(',')
    x.append(float(d[0]))
    y.append(float(d[1]))
    z.append(float(d[2]))
    #col.append(float(d[3]))
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


f = [(float(l.strip(' \r\n').split(',')[0])**2+float(l.strip(' \r\n').split(',')[1])**2)**0.5 for l in open('swissroll.csv').readlines()]
d = max(f)-min(f)
#print f
f = [[0,1.-(i-min(f))/(d),(i-min(f))/(d)] for i in f]

cs = np.array(f) #np.array([0.]*333+[0.25]*235+[0.5]*281+[0.75]*209) #range(len(x))) #np.array(col)
print len(cs),len(zs)
#ax.scatter(xs, ys,c=cs,s=20,cmap=matplotlib.cm.hot)

ax.scatter(xs, ys, zs, c=cs,s=20,cmap=matplotlib.cm.hot)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
