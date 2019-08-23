import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# X is the target, Y is the source
def nn_ind_mapping(X, Y, pca_reduce=3, normalize=True, finish_early_factor=0.03, max_iter=500, min_change=2):
    n = len(X)
    X = np.array(X)
    Y = np.array(Y)

    if X.shape[1:] != Y.shape[1:]:
        raise ValueError(f'X and Y must contain elements of the same dimensionality: {X[0].shape} vs {Y[0].shape}')

    if normalize:
        X_avg = np.mean(X, 0)
        Y_avg = np.mean(Y, 0)
        X_std = np.std(X, 0)
        Y_std = np.std(X, 0)

        X_std[X_std == 0] = 1
        Y_std[Y_std == 0] = 1

        X = (X - X_avg) / X_std
        Y = (Y - Y_avg) / Y_std

    pca = PCA(n_components=pca_reduce).fit(np.concatenate([X, Y]))
    X = pca.transform(X)
    Y = pca.transform(Y)

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
