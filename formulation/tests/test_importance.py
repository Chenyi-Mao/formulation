"""
This function performs a test on the function `importance`
"""
import unittest

import numpy as np
import pandas as pd

from ..modules.importance import importance


rawdata = pd.read_csv('./formulation/data/FDA_APPROVED.csv')
data = pd.DataFrame({'Unchanged_excretion_in_urine':
                     rawdata['% Excreted Unchanged in Urine'],
                     'cLogP': rawdata['CLogP'], 'HBA': rawdata['HBA'],
                     'HBD': rawdata['HBD'],
                     'PSDA': rawdata['PSDA'], 'Formulation':
                     rawdata['Formulation']})
data = data.dropna()


class test_importance(unittest.TestCase):

    def test_importance(self):
        """
        This function tests whether the sum of all importance is 1
        """

        a = importance(data[['Unchanged_excretion_in_urine',
                             'cLogP', 'HBA',
                             'HBD', 'PSDA']], data['Formulation'],
                       0.2, [100, 300, 500, 700, 1000])
        b = a.drop(columns='n')
        c = np.array(b)

        for i in range(len([100, 300, 500, 700, 1000])):
            assert np.round(np.sum(
                c[i]), 6) == 1, "Importance sum is not 1, something wrong"

        return
