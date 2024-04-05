
import pandas as pd
from sklearn.impute import SimpleImputer
import unittest
import numpy as np

class data_cleaner():
    def __init__(self,data):
        self.data = data
        self.imputer = SimpleImputer(strategy='median')

    
    def clean_data(self):
        data_num= self.data.select_dtypes([])
        self.imputer.fit(data_num)
        print(self.imputer.statistics_)
        return self.imputer.transform(data_num)
class TestCode(unittest.TestCase):

    def test_instance(self):
        imputer = SimpleImputer(strategy='median')

    def test_function(self):
        def clean_data(df):
            imputer = SimpleImputer(strategy='median')