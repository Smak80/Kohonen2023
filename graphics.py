import numpy as np
import math
from matplotlib import pyplot as plt

mrk = ['^', 's', 'p', 'h', 'o']

def get_colors(x):
    if len(x) > 0:
        cx = x[:, 0:3]
        cx = cx / 255
        cx[cx < 0] = 0
        cx[cx > 1] = 1
    else:
        cx = np.array([])
    return cx


def get_sizes(x):
    return x[:, -2]


def get_forms(x):
    return [mrk[fi] for fi in x[:, -1]]


def plot_data(data):
    sz = len(data)
    if sz == 0:
        return
    cdata = get_colors(data)
    sdata = np.array(get_sizes(data), dtype=float)
    mdata = get_forms(data)
    qsz = math.ceil(math.sqrt(sz))
    k = 0
    #xd = np.array([1 + int(i / qsz) for i in range(sz)])
    #yd = np.array([1 + i % qsz for i in range(sz)])
    for x in range(qsz):
        for y in range(qsz):
            plt.scatter(x, y, color=cdata[k], s=sdata[k]*50, marker=mdata[k])
            k += 1
            if k >= len(sdata):
                break
        if k >= len(sdata):
            break
    plt.show()
