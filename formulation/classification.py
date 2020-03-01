import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt


def get_datasets():
    df = pd.read_csv('table.csv') 
    print(df.head(3)) 
    print(df.columns)

    features = ['CLogP', 'HBA', 'HBD', 'PSDA']
    predicted_feature = ['Measured LogP']

    X_df = df[features]
    X_data = X_df.values
    print("X:", X_data.shape)

    y_df = df['Formulation']
    y_data = y_df.values
    print("y:", y_data.shape)

    # remove nan from X_data and corresponding rows in y_data
    X_rows = []
    for i in range(X_data.shape[0]):
        row = [str(x) for x in X_data[i,:]]
        if 'nan' not in row:
            X_rows.append(i)

    X_data = X_data[X_rows,:]
    print("X:", X_data.shape)

    y_data = y_data[X_rows]
    print("y:", y_data.shape)

    # remove nan from y_data and corresponding rows in X_data
    y_rows = []
    for i in range(y_data.shape[0]):
        if str(i) != 'nan':
            y_rows.append(i)

    X_data = X_data[y_rows,:]
    print("X:", X_data.shape)

    y_data = y_data[y_rows]
    print("y:", y_data.shape)


    # select three classes
    forms = ['solution', 'capsules', 'tablets']
    y_rows = []
    labels = []
    for (i, each) in enumerate(y_data):
        form = str(each).split(' ')[0].strip(',')
        if form in forms:
            y_rows.append(i)
            labels.append(form)

    X = X_data[y_rows,:]
    print("X:", X_data.shape)

    # y = y[y_rows]
    # print("y:", y.shape)
    print("labels:", set(labels))
    print("labels:", len(labels))

    # transform class labels into numeric values
    y = []
    for each in labels:
        if each == forms[0]:
            y.append(1)
        if each == forms[1]:
            y.append(2)
        if each == forms[2]:
            y.append(3)
    y = np.array(y)
    print(y.shape)

    print("X:", X.shape)
    print("y:", y.shape)

    return X, y


if __name__ == '__main__':

    X, y = get_datasets()
    d = X.shape[1]; n = X.shape[0]
    dataset = np.concatenate([X, y.reshape(-1,1)], axis = -1)

    Ntst = n//10
    print("Ntst:", Ntst)

    # 10-fold cross validation to choose best max_depth of tree
    np.random.shuffle(dataset)
    X = dataset[:,:d]
    y = dataset[:,-1]

    max_depths = range(1,100)
    best_accuracy = 0.
    best_depth = 0
    accuracies_k_fold = []

    for max_depth in max_depths:

        accuracies = []
        for k in range(10-1):

            i = k*Ntst
            j = (k+1)*Ntst

            X_tst = X[i:j,:]
            y_tst = y[i:j]

            if i==0:
                X_trn = X[j:,:]
                y_trn = y[j:]
            else:
                X_trn = np.concatenate([X[:i,:], X[j:,:]], axis = 0)
                y_trn = np.concatenate([y[:i], y[j:]])

    
            clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
            clf.fit(X_trn,y_trn)
            # print(clf.feature_importances_)

            y_pred = clf.predict(X_tst)
            accuracy = [1 for i in range(Ntst) if y_pred[i]==y_tst[i]]
            accuracy = np.sum(accuracy)/Ntst
            # print("accuracy:", accuracy)
            accuracies.append(accuracy)

        acc_k_fold = np.mean(accuracies)
        accuracies_k_fold.append(acc_k_fold)
        if acc_k_fold > best_accuracy:
            best_accuracy = acc_k_fold
            best_depth = max_depth 


    plt.figure(figsize=(10,10))
    plt.plot(range(len(accuracies_k_fold)), accuracies_k_fold, 'r-*')
    plt.xlabel('Different max_depth')
    plt.ylabel('Accuracy on test dataset')
    plt.savefig('accuracy.png')

    print("best depth:", best_depth)
    print("best accuracy:", best_accuracy)








