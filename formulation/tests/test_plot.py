import os
from formulation.modules.plot import plot_figures


def test_plot_figures():

    fname = 'gridvalues.npy'
    assert os.path.exists(fname)

    plot_figures(fname)

    assert os.path.exists('grid.png')
    assert os.path.exists('scatter.png')


test_plot_figures()
