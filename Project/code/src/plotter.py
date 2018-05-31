from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math   # This will import math module

from matplotlib import gridspec


def plotModels(dataSet, labels, titles, N, n):
    fig = plt.figure()
    gs = gridspec.GridSpec(int(math.sqrt(N)+1), int(math.sqrt(N)+1)) 

    numRows = 4
    numColumns = 4
    for i in range(0, N):
        pts = dataSet[i]
        ax = plt.subplot(gs[i], projection = '3d')
        ax.scatter(dataSet[i][0:n, [0]], dataSet[i][0:n, [1]], dataSet[i][0:n, [2]], s = 1)
        ax.set_axis_off()
        ax.set_title(titles[labels[i][0]])

    plt.tight_layout()

    plt.show()

def plotPtsArray(pts, n):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(pts[0:n, [0]], pts[0:n, [1]], pts[0:n, [2]], s = 1)
    ax.set_axis_off();
    plt.show()