import numpy as np

from graphics import plot_data

norm_coeff = []
lengths = []

cluster_radius = 6

dist = lambda a, b: np.sqrt(np.sum((a - b) ** 2))
fltr = lambda d: d <= cluster_radius


def nearest(inp, nrn, limited = False):
    try:
        l = [dist(inp, w) for w in nrn]
        if limited: l = list(filter(fltr, l))
        return np.array(l, dtype=float).argmin()
    except:
        return None


def get_vec(vals):
    max = np.max(vals)
    lengths.append(max)
    res = np.zeros((len(vals), int(max)))
    for i in range(len(res)):
        res[i, int(vals[i]) - 1] = 3
    return res


# нормировка исходных данных
def norm(x):
    tx = np.asarray(x, dtype=float).T
    for i in range(3):
        norm_coeff.append((1 / np.std(tx[i]), np.median(tx[i])))
        tx[i] = (tx[i] * norm_coeff[-1][0] - norm_coeff[-1][1] * norm_coeff[-1][0])
    x = tx.T[:, :3]
    x = np.append(x, get_vec(tx[3]), axis=1)
    x = np.append(x, get_vec(tx[4]), axis=1)
    return x


def denorm(x):
    tx = np.asarray(x, dtype=float).T
    for i in range(3):
        tx[i] = (tx[i] + norm_coeff[i][1] * norm_coeff[i][0]) / norm_coeff[i][0]
    x = tx.T[:, :3]
    x = np.append(x, get_vec(tx[3]), axis=1)
    x = np.append(x, get_vec(tx[4]), axis=1)
    return x


def train(data, nrn, epoches = 50):
    eta = 0.7
    d_eta = 0.05
    while eta > 0:
        for e in range(epoches):
            for v in data:
                winner_idx = nearest(v, nrn, True)
                if winner_idx is None:
                    nrn.append(v)
                else:
                    nrn[winner_idx] = list(np.array(nrn[winner_idx]) + eta * (np.array(v) - np.array(nrn[winner_idx])))
        eta -= d_eta


def predict(inp, nrn):
    k = len(nrn)
    cluster = [[] for i in range(k)]
    ninp = norm(inp)
    for i in range(len(inp)):
        r = nearest(ninp[i], nrn)
        cluster[r].append(inp[i])
    for i in range(len(cluster)):
        cluster[i] = np.array(cluster[i])
    return cluster


def show_clusters(clusters):
    for cluster in clusters:
        plot_data(cluster)

# исходные данные
data = np.asarray(np.loadtxt("data.txt"), dtype=int)
plot_data(data)
w = []
# нормируем
n_data = norm(data)
train(n_data, w)

clusters = predict(data, w)
show_clusters(clusters)
