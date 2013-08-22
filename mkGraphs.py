from subprocess import Popen
import shlex
import os

for i in xrange(3000):
    #Popen(shlex.split("python ploteigens3d.py unroll/embedding"+str(i)+".csv"))
    os.system("python ploteigens3d.py unroll/embedding"+str(i).zfill(4)+".csv")
