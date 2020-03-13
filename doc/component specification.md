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
* What it does: find out the best combination of n estimotor and maximum depth for a given data
* Inputs (with type information): initial values of n estimoar and maximum depth (list), raw data in terms of selected features (dataframe), raw data in terms of a prediction feature (series)
* Outputs (with type information): parameter combination to achive the best accuracy (tuple) 
* How it interacts with other components: it uses raw data coming from predict_missing_value and its output is adapted as parameters for the other components. 
