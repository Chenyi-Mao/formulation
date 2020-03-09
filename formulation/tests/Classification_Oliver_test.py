import pandas as pd
import numpy as np
from formulation.modules.Classification_Oliver import predict
from formulation.modules.Classification_Oliver import Choose_N_property_and_determine_new_accuracy


def test_predict():
    raw_data = pd.read_csv('../data/FDA_APPROVED.csv')
    picked_data = pd.DataFrame({'Unchanged_excretion_in_urine':raw_data['% Excreted Unchanged in Urine'], 
     'cLogP':raw_data['CLogP'], 'HBA':raw_data['HBA'], 'HBD': raw_data['HBD'], 
                            'PSDA':raw_data['PSDA'], 'Formulation':raw_data['Formulation']})
    pick_data = picked_data.dropna()
    X = pick_data[['Unchanged_excretion_in_urine','cLogP','HBA', 'HBD', 'PSDA']]
    Y = pick_data['Formulation']
    assert  round(predict(X,Y), 2) == 0.73

def test_Choose_N_property_and_determine_new_accuracy():
    raw_data = pd.read_csv('../data/FDA_APPROVED.csv')
    picked_data = pd.DataFrame({'Unchanged_excretion_in_urine':raw_data['% Excreted Unchanged in Urine'], 
     'cLogP':raw_data['CLogP'], 'HBA':raw_data['HBA'], 'HBD': raw_data['HBD'], 
                            'PSDA':raw_data['PSDA'], 'Formulation':raw_data['Formulation']})
    pick_data = picked_data.dropna()
    X = pick_data[['Unchanged_excretion_in_urine','cLogP','HBA', 'HBD', 'PSDA']]
    Y = pick_data['Formulation']
    assert round(test_Choose_N_property_and_determine_new_accuracy(4,X,Y),2) == 0.68
