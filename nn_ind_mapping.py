import numpy as np
from sklearn.neighbors import NearestNeighbors

# X is the target, Y is the source
def nn_ind_mapping(X, Y, normalize=True, finish_early_factor=0.03, max_iter=500, min_change=2):
    n = len(X)
    X = np.array(X)
    Y = np.array(Y)
    i = 0
    amt_left = [len(X)]
    finish_early = int(finish_early_factor * n)

    if X.shape[1:] != Y.shape[1:]:
        raise ValueError(f'X and Y must contain elements of the same dimensionality: {X[0].shape} vs {Y[0].shape}')
    # if X.shape[0] > Y.shape[0]:
    #     raise ValueError('X must have fewer data points than Y')

    if normalize:
        X_avg = np.mean(X, 0)
        Y_avg = np.mean(Y, 0)
        X_std = np.std(X, 0)
        Y_std = np.std(X, 0)

        X = (X - X_avg) / X_std
        Y = (Y - Y_avg) / Y_std

    neigh = NearestNeighbors(1)
    neigh.fit(Y)
    best_dists, best_inds = neigh.kneighbors(X, 1, return_distance=True)
    best_inds = best_inds.ravel()

    return best_inds

# if __name__ == '__main__':
#
#     n = 1000
#     m = 5
#     X = np.random.randint(0, 255, (n, m))
#     Y = np.random.randint(0, 255, (n * 2, m))
#     mapping = min_diff_pair_mapping(X,Y)
#
#     score = np.average(np.sum((X - Y[mapping]) ** 2, 1))
#     print(score)
