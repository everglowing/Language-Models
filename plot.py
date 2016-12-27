# Function that reads multiple paths and plots them all
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import spline
from config.arguments import plot_parser
from utils.strings import FILES

import matplotlib.pyplot as plt
plt.use('Agg')
import numpy as np
import os

def moving_average(a, n=1):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

args = plot_parser.parse_args()
file1 = os.path.join(args.save_dir, FILES[5])
file2 = os.path.join(args.save_dir2, FILES[5])

data = np.genfromtxt(file1, delimiter=",")
data2 = np.genfromtxt(file2,delimiter=",")
X = data[:,0]
Y = data[:,1]
#Z = data[:,2]
X2 = data2[:,0]
Y2 = data2[:,1]

X = moving_average(X, args.smoothing)
Y = moving_average(Y, args.smoothing)
X2 = moving_average(X2, args.smoothing)
Y2 = moving_average(Y2, args.smoothing)

plt.plot(X2,Y2,X,Y, linewidth=2.0)
plt.savefig('plot.png')

if args.show_fig:
    plt.show()