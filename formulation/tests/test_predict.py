import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ..modules.predict import predict
from ..modules.predict_missing_value import data_dropna


def test_predict():
    data_path = "./formulation/data/"
    data_fname = 'FDA_APPROVED.csv'

    # Read csv file
    df = pd.read_csv(data_path+data_fname)
    print(df.tail(3))

    # Dropnan
    columns = ['CLogP', 'HBA', 'HBD', 'PSDA', 'Formulation']
    df = data_dropna(df, needed_cols=columns, subset=columns)
    print(df.tail(10))

    # Print count of each category
    print(df.groupby('Formulation').count())

    # Prepare predictors and response variable
    features = ['CLogP', 'HBA', 'HBD', 'PSDA']
    X_df = df[features]

    target = ['Formulation']
    y_df = df[target]

    X_trn, X_tst, y_trn, y_tst = train_test_split(
                X_df.values, y_df.values, test_size=0.20, random_state=42)

    clf = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=2,
                                random_state=0)
    clf.fit(X_trn, y_trn.flatten())
    y_pred = predict(clf, X_tst)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape[0] == y_tst.shape[0]

    X_tst_df = pd.DataFrame(X_tst)
    y_pred2 = predict(clf, X_tst_df)
    assert isinstance(y_pred2, np.ndarray)
    assert y_pred2.shape[0] == y_tst.shape[0]


test_predict()
