import pandas as pd
from modules.cross_validate import handle_missing_values
from modules.cross_validate import cross_validate_grid_search
# from cross_validate import cross_validate_n_predictors


def test_cross_validate_grid_search():

    # data_path = "../data/"
    data_path = "./formulation/data/"
    data_fname = 'FDA_APPROVED.csv'

    # Read csv file
    df = pd.read_csv(data_path+data_fname)
    print(df.tail(3))

    # Extract columns needed
    columns = ['CLogP', 'HBA', 'HBD', 'PSDA', 'Formulation']
    df = df[columns]

    # Handle missing values in selected columns
    df = handle_missing_values(df)
    print(df.tail(10))

    # Print count of each category
    print(df.groupby('Formulation').count())

    # Prepare predictors and response variable
    features = ['CLogP', 'HBA', 'HBD', 'PSDA']
    X_df = df[features]

    target = ['Formulation']
    y_df = df[target]

    max_depth = [1, 2]
    ntrees = [100]

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


def cross_validate_n_predictors():

            
test_cross_validate_grid_search():

    # data_path = "../data/"
    data_path = "./formulation/data/"
    data_fname = 'FDA_APPROVED.csv'

    # Read csv file
    df = pd.read_csv(data_path+data_fname)
    print(df.tail(3))

    # Extract columns needed
    columns = ['CLogP', 'HBA', 'HBD', 'PSDA', 'Formulation']
    df = df[columns]

    # Handle missing values in selected columns
    df = handle_missing_values(df)
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

