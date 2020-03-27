import numpy as np
from sklearn.datasets import make_blobs


def sample_three_blobs(n):
    centers = np.array([[5, 5], [-5, 5], [0, -5]])
    st_devs = np.array([[1.0, 1.0], [0.2, 0.2], [3.0, 0.5]])
    labels = np.random.randint(0, 3, size=(n,), dtype='int32')
    x = np.random.randn(n, 2) * st_devs[labels] + centers[labels]
    return x.astype('float32')


def sample_four_blobs(n):
    centers = np.array([[5, 5], [5, -5], [-5, -5], [-5, 5]])
    st_devs = [1.0, 1.0, 1.0, 1.0]
    x, _ = make_blobs(n, n_features=2, centers=centers, cluster_std=st_devs,
                      shuffle=True)
    return x.astype('float32')

def sample_smiley_data(n):
    count = n
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
       -np.sin(np.linspace(0, np.pi, count // 3))]

    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm].astype('float32'), data_y[perm]


def sample_diag_guass_data(count):
    rand = np.random.RandomState(0)
    return ([[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]).astype('float32')


def sample_cov_gauss_data(count):
    rand = np.random.RandomState(0)
    return ([[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])).astype('float32')
