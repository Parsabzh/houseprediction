from scipy.stats import binom
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#shows how to compute the 10.7% proba of getting a bad sample

sample_size = 1000
ratio_female = 0.511
proba_too_small = binom(sample_size, ratio_female).cdf(485-1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(535)
print(proba_too_small + proba_too_large)


# create income category
def split(data):
    data["income_cat"]= pd.cut(data["median_income"],bins=[0.,1.5,3.0,4.5,6.,np.inf],
                               labels=[1,2,3,4,5])
    train_set,test_set= train_test_split(data,test_size=0.2, random_state=42, stratify=data["income_cat"])
    data.drop('income_cat',axis= 'columns')
    return train_set, test_set

def split_x_y(data):
    X_data= data.drop("median_house_value",axis=1)
    y_data= data["median_house_value"]
    return X_data,y_data

