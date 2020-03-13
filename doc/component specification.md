# This component spcfification is written for all items in modules. 

## 1. predict_missing_value.py
* Function Name: drop_na
* What it does: screening columns of interests and delete empty cells
* Inputs (with type information): raw data (dataframe), names of select columns (list), names of column of interests (list)
* Outputs (with type information): cleaned raw data (dataframe)
* How it interacts with other components: cleaned raw data will be used as input for all other components. 

* Function Name: fill_missing_value
* What it does: use random forest regressor to predict missing values for columns of interests
* Inputs (with type information): raw data (dataframe), names of select columns (list), names of columns used as input for the regression (list), name of a column used as output for the regression (list)
* Outputs (with type information): a new raw data with filled value in the last argument (dataframe)
* How it interacts with other components: provide another raw data for users as input for all other components. 

## 2. cross_validation.py
* Function Name: cross_validate_grid_search
* What it does: find out the best combination of n estimotor and maximum depth for a given data
* Inputs (with type information): initial values of n estimoar and maximum depth (list), raw data in terms of selected features (dataframe), raw data in terms of a prediction feature (series)
* Outputs (with type information): parameter combination to achive the best accuracy (tuple) 
* How it interacts with other components: it uses raw data coming from predict_missing_value and its output is adapted as parameters for the other components. 

## 3. importance.py
* Function Name: importance
* What it does: find out the importance of each selected features for the prediction accuracy
* Inputs (with type information): raw data in terms of selected features (dataframe), raw data in terms of a prediction feature (dataframe), assigned test size (float), n estimator (float)
* Outputs (with type information): an importance report for each selected features
* How it interacts with other components: it uses raw data coming from predict_missing_value and n estimator determined by cross_validation. its outcome provides users finalize features before moving on to do a classification. 

## 4. classification.py
* Function Name: determine_new_accuracy
* What it does: re-calculate accuracies if users decide to choose partial features according to results from importance.py
* Inputs (with type information): raw data in terms of selected features (dataframe), raw data in terms of a prediction feature (dataframe), number of features (float)
* Outputs (with type information): accuracy at newly selected features
* How it interacts with other components: after checking importance.py, it provides users a flexibility to adjust features of interests. its outcome is one of paramenters that can be used to describe the model. 

* Function Name: classfication
* What it does: build random forest classifer model and use the model to make a prediction 
* Inputs (with type information): raw data applied to build the model (dataframe), data of interest for the prediction
* Outputs (with type information): the predicted formulation
* How it interacts with other components: it wraps up all py files above to build a model and then apply the model on test data to make a final prediction. 
