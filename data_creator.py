import numpy as np


def discr(X):
    X[X >= 0.75] = 4
    X[X < 0.25] = 1
    X[X < 0.5] = 2
    X[X < 0.75] = 3
    return X


cnt = 200
colors = np.asarray(np.random.random((cnt, 3))*255, dtype=int)
sizes = np.asarray(discr(np.random.normal(loc = 0.5, scale=0.3, size = (cnt, 1))), dtype=float)
forms = np.asarray(discr(np.random.random((cnt, 1))), dtype=float)
data = np.append(colors, sizes, 1)
data = np.append(data, forms, 1)
np.savetxt("data.txt", data)
