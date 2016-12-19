# Function that reads multiple paths and plots them all
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import spline
import numpy as np

def moving_average(a, n=1):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data = np.genfromtxt("graph.txt",delimiter=",")
data2 = np.genfromtxt("graph2.txt",delimiter=",")
X = data[:,0]
Y = data[:,1]
#Z = data[:,2]
X2 = data2[:,0]
Y2 = data2[:,1]

X = moving_average(X)
Y = moving_average(Y)
X2 = moving_average(X2)
Y2 = moving_average(Y2)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.contour(X, Y, Z)
plt.plot(X2,Y2,X,Y, linewidth=2.0)
plt.show()