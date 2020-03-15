import numpy as np
import pandas as pd


def predict(model, X_df):

    """
        This function takes in a trained model and test dataset
        to make predictions.

        Arguments:
        model: the trained random forest classifier
        X_tf: the input data samples for testing

        Return:
        The predicted classes for each input sample.

    """
    if isinstance(X_df, pd.DataFrame):
        X = X_df.values

    if isinstance(X_df, np.ndarray):
        X = X_df

    y_pred = model.predict(X)

    return y_pred
