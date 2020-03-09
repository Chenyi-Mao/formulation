import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def predict(X_data, Y_data):
    """
    X_data: is the user choice data utilized to prepare to analyze the result
    Y_data: is the desired result
    rtype: print out the result
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X_data, Y_data, test_size=0.1, random_state=1)
    RFC = RandomForestClassifier(n_estimators=100, random_state=2)
    Classicifation = RFC.fit(x_train, y_train)
    y_pred = Classicifation.predict(x_test)
    importance = pd.Series(Classicifation.feature_importances_,
                           index=list(X_data)).sort_values(ascending=False)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Feature Importance \n", importance)
    print("\n Classicifation report\n", classification_report(y_test, y_pred))
    print("The predict Classicifation for all \n", y_pred)
    return accuracy_score(y_test, y_pred)


# let user select first N's picks.
def Choose_N_property_and_determine_new_accuracy(N, X_data, Y_data):
    """
    pick the the most important N factors, let user pick 
    N: is the number of factors that users want to utilize
    X_data: The data need to be inputed 
    Y_data: The data need to be output
    return the new report based on the 
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X_data, Y_data, test_size=0.1, random_state=1)
    RFC = RandomForestClassifier(n_estimators=100, random_state=2)
    Classicifation = RFC.fit(x_train, y_train)
    y_pred = Classicifation.predict(x_test)
    importance_df = pd.DataFrame(Classicifation.feature_importances_,
                                 #index=['Unchanged_excretion_in_urine','cLogP','HBA', 'HBD', 'PSDA'],
                                 index=list(X_data), columns=['Value']).sort_values(by=['Value'], ascending=False)
    indexlist = []
    first_N_Picked = importance_df[:N]
    for i in range(N):
        indexlist.append(importance_df.index[i])
    # return first_N_Picked
    # return indexlist
    predict(X_data[indexlist], Y_data)
    return accuracy_score(y_test, y_pred)
