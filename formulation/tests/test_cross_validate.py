import pandas as pd
from ..modules.cross_validate import cross_validate_grid_search
from ..modules.cross_validate import cross_validate_n_predictors
from ..modules.predict_missing_value import data_dropna


def test_cross_validate_grid_search():

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

    max_depth = range(1, 5)
    ntrees = range(1, 200, 50)

    values = [max_depth, ntrees]
    results = cross_validate_grid_search(values, X_df, y_df)

    assert len(results) == 4
    assert len(results[0]) == 2
    assert len(results[1]) == 2
    assert len(results[2]) == 2
    assert len(results[3]) == 2
    m = len(results)
    n = len(results[0])
    for i in range(m):
        for j in range(n):
            assert isinstance(results[i][j], int)


def test_cross_validate_n_predictors():

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

    max_depth = 2
    n_estimators = 100

    best_p = cross_validate_n_predictors(X_df, y_df, max_depth, n_estimators)

    assert isinstance(best_p, int)
    assert best_p <= X_df.shape[1]


test_cross_validate_grid_search()
test_cross_validate_n_predictors()
