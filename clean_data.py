
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
        combined_data = pd.concat([self.X_train_data, self.X_test_data], axis=0)
        data_num = combined_data.select_dtypes(include=[np.number])
        self.imputer.fit(data_num)
        X_train= self.imputer.transform(self.X_train_data.select_dtypes(include=[np.number]))
        X_test= self.imputer.transform(self.X_test_data.select_dtypes(include=[np.number]))
        print(self.imputer.statistics_)
        return X_train, X_test
    
    def to_one_hot(self):
        if "ocean_proximity" in  self.X_train_data.columns and "ocean_proximity" in self.X_test_data.columns:
            train=self.cat_encoder.fit_transform(self.X_train_data["ocean_proximity"])
            test= self.cat_encoder.transform(self.X_test_data["ocean_proximity"])
            print (self.cat_encoder.categories_)
            print(self.cat_encoder.feature_names_in_)
            return train, test
        else:
            print("ocean_proximity not in train and test data")
            return None, None
            