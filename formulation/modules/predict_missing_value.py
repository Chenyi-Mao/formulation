"""
This is a submodule that use random forest regression to fill missing data.
"""
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def data_dropna(data, needed_cols='all', subset='all'):
    """
    This function selects columns specified by user and drops all rows with
    any NA value that is considered a null value by pandas.isnull()
    (pd.NA, np.nan, None...)

    # Arguments
        data: A pandas.DataFrame. The Input data.
        needed_cols: Array-like. The columns to be selected. Default `all`.

    # Returns
        The cleaned data after selecting and dropping null values.
    """

    assert isinstance(data, pd.DataFrame),\
        "Incorrect data type. Only pandas.DataFrames are accepted."
    assert isinstance(needed_cols, (list, str)),\
        "Incorrect data type. Only lists are accepted."
    assert all(col in needed_cols for col in subset),\
        "Error. `subset` contains element that is not in `needed_cols`."

    if needed_cols == 'all':
        needed_cols = list(data)
    if subset == 'all':
        subset = needed_cols

    return data[needed_cols].dropna(subset=subset)


def fill_missing_value(data, needed_cols, train_inputs, train_outpus,
                       test_size=0.2, n_estimators=100, max_depth=8):
    """
    This function use the features (train_inputs) and labels (train_outputs) to
    train a random forest regressor to predict missing values in labels.

    # Arguments
        data: A pandas.DataFrame. The Input data.
        needed_cols: Array-like. The columns to be selected.
        train_inputs: A list. The column indices used as features.
        train_outpus: A list. The column indices used as labels.
        test_size: A float number between 0.0 and 1.0. the proportion of the
                   dataset to include in the train split.
        n_estimators: An int. The number of trees in the random forest
                      regressor.
        max_depth: An integer, default is 8. The maximum depth of the tree.

    # Returns
        A dataset with missing values in data[train_outputs] filled as much as
        possible.
    """
    assert isinstance(test_size, float), "Test_size should be a float number"
    assert 0 <= test_size <= 1.0, "Test_size should be in the range of 0 to 1."
    assert isinstance(n_estimators, int), "n_estimators should be an integer."
    assert n_estimators > 0, "n_estimators should be positive."
    assert isinstance(max_depth, (int, type(None))), "max_depth should be an integer or None."
    assert max_depth == None or max_depth > 0, "max_depth should be positive."

    clean_data = data_dropna(data, needed_cols, needed_cols)

    # Split data into training and testing sets
    train, test = train_test_split(clean_data, test_size=test_size, random_state=2)
    x_train = train[train_inputs]
    y_train = train[train_outpus]
    x_test = test[train_inputs]
    y_test = test[train_outpus]

    regressor = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth)
    regressor.fit(x_train, y_train)

    cod = regressor.score(x_test, y_test)
    mse = mean_squared_error(regressor.predict(x_test), y_test)

    print("Coefficient of determination on testing set: {:.2f}".format(cod))
    print("Mean squared error on testing set: {:.2f}".format(mse))

    inputs_clean_data = data_dropna(data, needed_cols, train_inputs)
    filled_data = pd.DataFrame(columns=list(inputs_clean_data))

    for _, row in data.iterrows():
        if pd.isna(row[train_outpus]):
            if True not in pd.isna(np.array(row[train_inputs])):
                inputs = row[train_inputs].to_numpy().reshape(1, -1)
                row[train_outpus] = regressor.predict(inputs)[0]
        else:
            pass

        filled_data = filled_data.append(row)

    return filled_data
