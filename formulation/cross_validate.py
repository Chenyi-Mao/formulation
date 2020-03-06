import numpy as np 
import math
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

def handle_missing_values(df):
    # filling missing value using fillna()   
    # df = df.fillna(0) 


    # filling a missing value with 
    # previous ones   
    # df = df.fillna(method ='pad') 

    # filling  null value using fillna() function   
    # df = df.fillna(method ='bfill') 


    # using dropna() function   
    df = df.dropna() 

    return df 

# cross validation to choose max_depth of trees and number of trees
def cross_validate_grid_search(values, X, y):
    # values is a list of two lists [[different max_depth],[different ntrees]]

    assert X.shape[0]==y.shape[0]

    best_accuracy = 0.
    best_depth = 0
    best_ntrees = 0
    accuracies_k_fold = []

    solution_k_fold=[]
    capsules_k_fold=[]
    tablets_k_fold=[]

    grid = []


    for max_depth in values[0]:
        for ntrees in values[1]:
            grid.append([max_depth, ntrees])

            accuracies = []
            solution=0.
            capsules=0.
            tablets=0.


            kf = KFold(n_splits=10, shuffle=True, random_state=123)
            for train_index, test_index in kf.split(X):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_trn, X_tst = X[train_index], X[test_index]
                y_trn, y_tst = y[train_index], y[test_index]

                clf = RandomForestClassifier(n_estimators=ntrees, max_depth=max_depth, random_state=0)
                clf.fit(X_trn,y_trn)

                y_pred = clf.predict(X_tst)
                accuracy = [1 for i in range(y_pred.shape[0]) if y_pred[i]==y_tst[i]]
                accuracy = np.sum(accuracy)/y_pred.shape[0]
                accuracies.append(accuracy)


                report = classification_report(y_tst, y_pred, output_dict=True,zero_division=1)
                solution+=report['solution']['f1-score']
                capsules+=report['capsules']['f1-score']
                tablets+=report['tablets']['f1-score']

            acc_k_fold = np.mean(accuracies)
            accuracies_k_fold.append(acc_k_fold)

            solution_k_fold.append(solution/10)
            capsules_k_fold.append(capsules/10)
            tablets_k_fold.append(tablets/10)

            if acc_k_fold > best_accuracy:
                best_accuracy = acc_k_fold
                best_depth = max_depth 
                best_ntrees = ntrees

            print('max depth: {:d}, n_estimators: {:d}, accuracy: {:f}'.format(max_depth,ntrees,acc_k_fold))


    # print(np.argmax(solution_k_fold))
    # print(np.argmax(capsules_k_fold))
    # print(np.argmax(tablets_k_fold))

    best_for_solution = grid[np.argmax(solution_k_fold)]
    best_for_capsules = grid[np.argmax(capsules_k_fold)]
    best_for_tablets = grid[np.argmax(tablets_k_fold)]

    best_solution_accuracy = solution_k_fold.max()
    best_capsules_accuracy = capsules_k_fold.max()
    best_tablets_accuracy = tablets_k_fold.max()

    print('Best accuracy for solution: {:f}'.formt(best_solution_accuracy))
    print('Best accuracy for capsules: {:f}'.formt(best_capsules_accuracy))
    print('Best accuracy for tablets: {:f}'.formt(best_tablets_accuracy))

    print('Best accuracy for total: {:f}'.formt(best_accuracy))

    return best_depth, best_ntrees, best_for_solution, best_for_capsules, best_for_tablets


# cross validation to choose number of predictors to use
def cross_validate_n_predictors(X, y, max_depth, n_estimators):
    assert X.shape[0]==y.shape[0]

    p = X.shape[1]
    p_values = range(1,p+1)

    best_accuracy = 0.
    best_p = p
    accuracies_k_fold = []

    for p_val in p_values:

        accuracies = []

        kf = KFold(n_splits=10, shuffle=True, random_state=123)
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_trn, X_tst = X[train_index,:p_val], X[test_index,:p_val]
            y_trn, y_tst = y[train_index], y[test_index]

            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, random_state=0)
            clf.fit(X_trn,y_trn)

            y_pred = clf.predict(X_tst)
            accuracy = [1 for i in range(y_pred.shape[0]) if y_pred[i]==y_tst[i]]
            accuracy = np.sum(accuracy)/y_pred.shape[0]
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
    X = df[features].values
    
    target = ['Formulation']
    y = df[target].values
    y = y.flatten()


    max_depth = range(1,10)
    ntrees = [5, 10, 20, 50, 100, 200, 500]
    # max_depth = [1,2]
    # ntrees = [100]

    best_depth, best_ntrees, best_for_solution, best_for_capsules, best_for_tablets = cross_validate_grid_search([max_depth, ntrees], X, y)
    print('Best max_depth: {:d}, best n_estimators: {:d}'.format(best_depth, best_ntrees))
    print('Best parameter combination for solution catogory:', best_for_solution)
    print('Best parameter combination for capsules catogory:', best_for_capsules)
    print('Best parameter combination for tablets catogory:', best_for_tablets)

    best_p = cross_validate_n_predictors(X, y, best_depth, best_ntrees)

    # Best max_depth: 9, best n_estimators: 100
    # Best parameter combination for solution catogory: [6, 20]
    # Best parameter combination for capsules catogory: [9, 5]
    # Best parameter combination for tablets catogory: [9, 100]
    # p: 1, accuracy: 0.552941
    # p: 2, accuracy: 0.552941
    # p: 3, accuracy: 0.635294
    # p: 4, accuracy: 0.623529

    






