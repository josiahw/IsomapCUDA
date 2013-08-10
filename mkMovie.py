from subprocess import Popen
import shlex
import time
import os

#s = Popen(shlex.split("python mkGraphs.py && convert -delay 6 -quality 95 unroll/embedding*png embedding.mpg"))
os.system("python mkGraphs.py && ffmpeg -qscale 5 -r 20 -b 9600 -i unroll/embedding%4d.png movie.mp4")
time.sleep(0.01)
