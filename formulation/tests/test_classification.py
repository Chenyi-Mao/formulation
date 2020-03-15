import pandas as pd
from ..modules.classification import predict
from ..modules.classification import determine_new_accuracy


def test_predict():
    raw_data = pd.read_csv('./formulation/data/FDA_APPROVED.csv')
    picked_data = pd.DataFrame({'Unchanged_excretion_in_urine':
                                raw_data['% Excreted Unchanged in Urine'],
                                'cLogP': raw_data['CLogP'],
                                'HBA': raw_data['HBA'], 'HBD': raw_data['HBD'],
                                'PSDA': raw_data['PSDA'],
                                'Formulation': raw_data['Formulation']})
    pick_data = picked_data.dropna()
    X = pick_data[['Unchanged_excretion_in_urine',
                   'cLogP', 'HBA', 'HBD', 'PSDA']]
    Y = pick_data['Formulation']

    assert not isinstance(predict(X, Y), float), "wrong type 000"


def test_determine_new_accuracy():
    raw_data = pd.read_csv('./formulation/data/FDA_APPROVED.csv')
    picked_data = pd.DataFrame({'Unchanged_excretion_in_urine':
                                raw_data['% Excreted Unchanged in Urine'],
                                'cLogP': raw_data['CLogP'],
                                'HBA': raw_data['HBA'], 'HBD': raw_data['HBD'],
                                'PSDA': raw_data['PSDA'],
                                'Formulation': raw_data['Formulation']})
    pick_data = picked_data.dropna()
    X = pick_data[['Unchanged_excretion_in_urine',
                   'cLogP', 'HBA', 'HBD', 'PSDA']]
    Y = pick_data['Formulation']
    assert isinstance(determine_new_accuracy(
        3, X, Y), float), "wrong type 111"


test_predict()
test_determine_new_accuracy()
