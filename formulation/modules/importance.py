"""
This is a submodule to evaluate the importance
of each predictor in Random Forest Classifier
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def importance(X_data, y_data, test_size, nlist):
    """
    This function perform a Random Forest Classifier Fit
    on X_train data and y_train data,through a list of n_estimators,
    to evaluate the importance of each predictor in classification.

    #Arguments
        X_data: The data regarded as predictors in classification
        y_data: The data regarded as response in classification
        test_size: The test data size in training and testing data spliting
        nlist: A list containing several integers
        treated as parameter `n_estimators`

    #returns
        A dataframe composed of importance of each predictor
         with correspinding n_estimator for a specific test size
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size)

    nv = np.zeros((len(nlist), 1))
    importance = np.zeros((len(nlist), X_data.shape[1]))
    imp = np.zeros((len(nlist), X_data.shape[1]+1))

    i = 0
    for n in nlist:
        nv[i] = n
        RF = RandomForestClassifier(n_estimators=n)
        Class = RF.fit(X_train, y_train)
        importance[i] = Class.feature_importances_
        imp[i] = np.hstack((nv[i], importance[i]))
        cols = ['n']+list(X_data.columns)
        i = i+1
    im = pd.DataFrame(imp, columns=cols)
    return im
