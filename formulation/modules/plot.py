import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def plot_figures(fname):
    """
        This function takes in the generated (gridpoints, values)
        from cross_validate.cross_validate_grid_search function,
        and produces a gird figure and a scatter plot.

        Arguments:
        fname: the file name of the generated (gridpoints, values)
        array

        No retured value.
    """
    arr = np.load(fname)
    n = arr.shape[0]

    # Construct grid of range [x_lb, x_ub] X [y_lb, y_ub]
    x_lb = arr[:, 0].min()
    x_ub = arr[:, 0].max()
    y_lb = arr[:, 1].min()
    y_ub = arr[:, 1].max()

    nm_arr = arr
    nm_arr[:, 0] = (arr[:, 0] - x_lb) / (x_ub - x_lb)
    nm_arr[:, 1] = (arr[:, 1] - y_lb) / (y_ub - y_lb)

    n_x, n_y = n, n
    x = np.linspace(0., 1., n_x)
    y = np.linspace(0., 1., n_y)
    grid_x, grid_y = np.meshgrid(x, y)

    # Interpolate through values at points over grid
    grid = griddata(arr[:, :2], arr[:, -1], (grid_x, grid_y), method='cubic')

    # Plot grid figure
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.T, extent=(0, 1, 0, 1), origin='lower')
    plt.scatter(arr[:, 0], arr[:, 1], marker='o', s=5, zorder=10)
    plt.colorbar()
    plt.xlabel('Max_depth from range({:d},{:d}) normalized to [0,1]'.format(
                int(x_lb), int(x_ub)))
    plt.ylabel('N_estimators from range({:d},{:d}) normalized to [0,1]'.format(
                int(y_lb), int(y_ub)))
    plt.title('Accuracy vs. (max_depth, n_estimators)')
    plt.savefig('grid.png')

    # Plot scatter figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=Axes3D.name)
    xs = arr[:, 2]
    ys = arr[:, 0]
    zs = arr[:, 1]
    ax.scatter(xs, ys, zs, marker='o')

    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Max_depth', fontsize=12)
    ax.set_zlabel('N_estimators', fontsize=12)
    plt.title(
        'Grid search for values of max_dpeth and n_estimators.', fontsize=12)
    plt.savefig('scatter.png')

if __name__ == '__main__':
    fname = '../tests/gridvalues.npy'
    plot_figures(fname)
