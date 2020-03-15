import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from formulation.modules.predict_missing_value import data_dropna
from formulation.modules.predict import predict


def cross_validate_grid_search(values, X_df, y_df):

    """
        This function uses 10-fold cross-validation to choose the best
        combination of max_depth and n_estimators for random forest classifier.

        Arguments:
        values: [list1, list2], where list1 is a list of values for max_depth
        and list2 is a list of values for n_estimators

        X_df: dataframe, where each column represents a predictor
        y_df: series, representing the response variable

        Return:
        best_for_total: tuple, parameter combination to achieve the highest
                        total accuracy
        best_for_solution: tuple, parameter combination to achive the highest
                        accuracy for classifing "solution"
        best_for_capsules: tuple, parameter combination to achive the highest
                        accuracy for classifing "capsules"
        best_for_tablets: tuple, parameter combination to achive the highest
                        accuracy for classifing "tablets"

    """
    assert len(values) == 2
    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(y_df, pd.DataFrame)

    X = X_df.values
    y = y_df.values.flatten()

    assert X.shape[0] == y.shape[0]

    best_accuracy = 0.
    best_depth = 0
    best_ntrees = 0
    accuracies_k_fold = []

    solution_k_fold = []
    capsules_k_fold = []
    tablets_k_fold = []

    grid = []
    gridvalues = []

    for max_depth in values[0]:
        for ntrees in values[1]:
            grid.append([max_depth, ntrees])

            accuracies = []
            solution = 0.
            capsules = 0.
            tablets = 0.

            kf = KFold(n_splits=10, shuffle=True, random_state=123)
            for train_index, test_index in kf.split(X):
                X_trn, X_tst = X[train_index], X[test_index]
                y_trn, y_tst = y[train_index], y[test_index]

                clf = RandomForestClassifier(
                                    n_estimators=ntrees,
                                    max_depth=max_depth,
                                    random_state=0)
                clf.fit(X_trn, y_trn)

                y_pred = clf.predict(X_tst)
                m = y_pred.shape[0]
                accuracy = [1 for i in range(m) if y_pred[i] == y_tst[i]]
                accuracy = np.sum(accuracy)/m
                accuracies.append(accuracy)

                report = classification_report(
                    y_tst, y_pred, output_dict=True, zero_division=1)
                solution += report['solution']['f1-score']
                capsules += report['capsules']['f1-score']
                tablets += report['tablets']['f1-score']

            acc_k_fold = np.mean(accuracies)
            accuracies_k_fold.append(acc_k_fold)

            solution_k_fold.append(solution/10)
            capsules_k_fold.append(capsules/10)
            tablets_k_fold.append(tablets/10)

            if acc_k_fold > best_accuracy:
                best_accuracy = acc_k_fold
                best_depth = max_depth
                best_ntrees = ntrees

            print('max depth: {:d}, n_estimators: {:d}, accuracy: {:f}'.format(
                max_depth, ntrees, acc_k_fold))
            gridvalues.append([max_depth, ntrees, acc_k_fold])

    # End search
    # Save grid points and values for plotting figures
    np.save('../tests/gridvalues.npy', gridvalues)

    best_for_total = (best_depth, best_ntrees)

    best_for_solution = grid[np.argmax(solution_k_fold)]
    best_for_capsules = grid[np.argmax(capsules_k_fold)]
    best_for_tablets = grid[np.argmax(tablets_k_fold)]

    best_solution_accuracy = np.max(solution_k_fold)
    best_capsules_accuracy = np.max(capsules_k_fold)
    best_tablets_accuracy = np.max(tablets_k_fold)

    print('Best accuracy for solution: {:f}'.format(best_solution_accuracy))
    print('Best accuracy for capsules: {:f}'.format(best_capsules_accuracy))
    print('Best accuracy for tablets: {:f}'.format(best_tablets_accuracy))

    print('Best accuracy for total: {:f}'.format(best_accuracy))

    results = [
            best_for_total,
            best_for_solution,
            best_for_capsules,
            best_for_tablets]

    return results


def cross_validate_n_predictors(X_df, y_df, max_depth, n_estimators):

    """
        This function uses 10-fold cross-validation to choose the best
        number of predictors to be included in the training set.

        Arguments:
        X_df: dataframe, where each column represents a predictor
        y_df: series, representing the response variable

        max_depth: the maximum depth of the tree
        n_estimators: the number of trees in the forest.

        Return:
        best_p: number of predictors to be included to achieve
        the highest total accuracy
    """
    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(y_df, pd.DataFrame)
    assert isinstance(max_depth, int)
    assert isinstance(n_estimators, int)

    X = X_df.values
    y = y_df.values.flatten()

    assert X.shape[0] == y.shape[0]

    p = X.shape[1]
    p_values = range(1, p+1)

    best_accuracy = 0.
    best_p = p
    accuracies_k_fold = []

    for p_val in p_values:

        accuracies = []

        kf = KFold(n_splits=10, shuffle=True, random_state=123)
        for train_index, test_index in kf.split(X):
            X_trn, X_tst = X[train_index, :p_val], X[test_index, :p_val]
            y_trn, y_tst = y[train_index], y[test_index]

            clf = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=0)
            clf.fit(X_trn, y_trn)

            # y_pred = clf.predict(X_tst)
            y_pred = predict(clf, X_tst)
            m = y_pred.shape[0]
            accuracy = [1 for i in range(m) if y_pred[i] == y_tst[i]]
            accuracy = np.sum(accuracy)/m
            accuracies.append(accuracy)

        acc_k_fold = np.mean(accuracies)
        accuracies_k_fold.append(acc_k_fold)
        if acc_k_fold > best_accuracy:
            best_accuracy = acc_k_fold
            best_p = p_val

        print('p: {:d}, accuracy: {:f}'.format(p_val, accuracy))

    print('best p: {:d}, best accuracy: {:f}'.format(best_p, best_accuracy))

    return best_p


if __name__ == '__main__':

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
    X = df[features]

    target = ['Formulation']
    y = df[target]

    max_depth = range(1, 5)
    ntrees = range(1, 200, 50)

    results = cross_validate_grid_search([max_depth, ntrees], X, y)
    best_for_total = results[0]
    best_for_solution = results[1]
    best_for_capsules = results[2]
    best_for_tablets = results[3]
    print('Best max_depth: {:d}, best n_estimators: {:d}'.format(
                        best_for_total[0], best_for_total[1]))
    print('Best parameter for solution catogory:', best_for_solution)
    print('Best parameter for capsules catogory:', best_for_capsules)
    print('Best parameter for tablets catogory:', best_for_tablets)

    best_p = cross_validate_n_predictors(
                X, y, 2, 100)
