import os
from formulation.modules.plot import plot_figures


def test_plot_figures():

    fname = './formulation/tests/gridvalues.npy'
    assert os.path.exists(fname)

    plot_figures(fname)

    assert os.path.exists('./formulation/tests/grid.png')
    assert os.path.exists('./formulation/tests/scatter.png')


test_plot_figures()
