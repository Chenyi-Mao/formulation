import numpy as np 
import pandas as pd 
from formulation.modules.cross_validate import *

def test_cross_validate_grid_search():

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

	max_depth = [1,2]
	ntrees = [100]

	values = [max_depth, ntrees]


	assert len(cross_validate_grid_search(values, X_df, y_df))==4
	assert len(cross_validate_grid_search(values, X_df, y_df)[0])==2
	assert len(cross_validate_grid_search(values, X_df, y_df)[1])==2
	assert len(cross_validate_grid_search(values, X_df, y_df)[2])==2
	assert len(cross_validate_grid_search(values, X_df, y_df)[3])==2
	m = len(cross_validate_grid_search(values, X_df, y_df))
	n = len(cross_validate_grid_search(values, X_df, y_df)[0])
	for i in range(m):
		for j in range(n):
			assert type(cross_validate_grid_search(values, X_df, y_df)[i][j]) == type(1)

test_cross_validate_grid_search()
