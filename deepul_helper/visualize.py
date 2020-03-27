import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import make_grid


def plot_hist(data, bins=10, xlabel='x', ylabel='Probability', title='', density=None):
    bins = np.concatenate((np.arange(bins) - 0.5, [bins - 1 + 0.5]))

    plt.figure()
    plt.hist(data, bins=bins, density=True)

    if density:
        plt.plot(density[0], density[1], label='distribution')
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_2d_dist(dist, title='Learned Distribution'):
    plt.figure()
    plt.imshow(dist)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x0')
    plt.show()


def plot_train_curves(epochs, train_losses, test_losses, title=''):
    x = np.linspace(0, epochs, len(train_losses))
    plt.figure()
    plt.plot(x, train_losses, label='train_loss')
    if test_losses:
        plt.plot(x, test_losses, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_scatter_2d(points, title='', labels=None):
    plt.figure()
    if labels is not None:
        plt.scatter(points[:, 0], points[:, 1], c=labels,
                    cmap=mpl.colors.ListedColormap(['red', 'blue', 'green', 'purple']))
    else:
        plt.scatter(points[:, 0], points[:, 1])
    plt.title(title)
    plt.show()


def visualize_batch(batch_tensor, nrow=8, title='', figsize=None):
    batch_tensor = batch_tensor.clamp(min=0, max=1)
    grid_img = make_grid(batch_tensor, nrow=nrow)
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()