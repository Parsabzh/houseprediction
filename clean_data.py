
import pandas as pd
from sklearn.impute import SimpleImputer
import unittest
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config
from sklearn.ensemble import IsolationForest

class data_cleaner():
    def __init__(self,X_train_data,X_test_data):
        self.X_train_data = X_train_data
        self.X_test_data = X_test_data
        self.imputer = SimpleImputer(strategy='median')
        self.cat_encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        set_config(transform_output="pandas")
    def impute_data(self):
        data_num= self.data.select_dtypes([])
        self.imputer.fit(data_num)
        print(self.imputer.statistics_)
        return self.imputer.transform(data_num)
    
    def to_one_hot(self):

        train=self.cat_encoder.fit_transform(self.X_train_data["ocean_proximity"])
        test= self.cat_encoder.transform(self.X_test_data["ocean_proximity"])
        print (self.cat_encoder.categories_)
        print(self.cat_encoder.feature_names_in_)
        return train, test
    
class TestCode(unittest.TestCase):

    def test_instance(self):
        imputer = SimpleImputer(strategy='median')

    def test_function(self):
        def clean_data(df):
            imputer = SimpleImputer(strategy='median')
