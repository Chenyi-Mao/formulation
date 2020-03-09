"""
Test cases for predict_missing_value submodule.
"""
import pandas as pd

from formulation.modules.predict_missing_value import data_dropna
from formulation.modules.predict_missing_value import fill_missing_value


# Global variables used in each test
DATA = pd.read_csv("./formulation/data/FDA_APPROVED.csv")
NEEDED = ["MW Drug", "MW Sol", "CLogP", "HBA", "HBD", "PSA", "Measured LogP"]
INPUTS = NEEDED[:-1]
OUTPUT = NEEDED[-1]


def test_data_dropna():
    """
    Test case for `data_dropna`
    """

    # test handling illegal inputs
    try:
        # `data` should be a pd.DataFrame
        data_dropna([1, 2, None])
        raise Exception("Illegal input test failed!")
    except AssertionError:
        pass

    try:
        # `subset` contains elements that are not in `needed_cols`
        data_dropna(DATA, INPUTS, NEEDED)
        raise Exception("Illegal input test failed!")
    except AssertionError:
        pass

    all_dropped = data_dropna(DATA)
    inputs_dropped = data_dropna(DATA, NEEDED, INPUTS)
    needed_dropped = data_dropna(DATA, NEEDED, NEEDED)

    assert isinstance(all_dropped, pd.DataFrame),\
        "Type error. pd.DataFrame expected"
    assert isinstance(inputs_dropped, pd.DataFrame),\
        "Type error. pd.DataFrame expected"
    assert isinstance(needed_dropped, pd.DataFrame),\
        "Type error. pd.DataFrame expected"

    assert True not in all_dropped.isna().to_numpy(),\
        "Error. Missing value not dropped"
    assert True not in inputs_dropped[INPUTS].isna().to_numpy(),\
        "Error. Missing value not dropped"
    assert True not in needed_dropped.isna().to_numpy(),\
        "Error. Missing value not dropped"


def test_fill_missing_value():
    """
    Test case for `fill_missing_value`
    """
    try:
        # test_size need to be a positive float number
        fill_missing_value(DATA, NEEDED, INPUTS, OUTPUT, test_size=-1.0)
        raise Exception("Illegal input test failed!")
    except AssertionError:
        pass

    try:
        # n_estimators need to be a positive integer
        fill_missing_value(DATA, NEEDED, INPUTS, OUTPUT, n_estimators=-1)
    except AssertionError:
        pass

    try:
        # max_depth need to be a positive integer
        fill_missing_value(DATA, NEEDED, INPUTS, OUTPUT, max_depth=1.2)
    except AssertionError:
        pass

    filled_data = fill_missing_value(DATA, NEEDED, INPUTS, OUTPUT)

    assert isinstance(filled_data, pd.DataFrame),\
        "Type error. pd.DataFrame expected"


test_data_dropna()
test_fill_missing_value()
