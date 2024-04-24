
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler, FunctionTransformer
from sklearn import set_config
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
        self.imputer_cat= SimpleImputer(strategy="most_frecuent")
    def column_ratio(self, X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(self, function_transformer, feature_names_in):
        return ["ratio"]  # feature names out

    def ratio_pipeline(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(self.column_ratio, validate=False),  # feature_names_out should be validated elsewhere
            StandardScaler()
        )

    def log_pipeline(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(np.log, validate=False),
            StandardScaler()
        )

    def preprocess(self, X):
        ratio_pipe = self.ratio_pipeline()
        log_pipe = self.log_pipeline()
        cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
        cat_pipeline = make_pipeline(self.imputer_cat,self.one_hot_encoder)
        default_num_pipeline = make_pipeline(self.imputer, self.scaler)
        
        preprocessing = ColumnTransformer([
                ("bedrooms", ratio_pipe, ["total_bedrooms", "total_rooms"]),
                ("rooms_per_house", ratio_pipe, ["total_rooms", "households"]),
                ("people_per_house", ratio_pipe, ["population", "households"]),
                ("log", log_pipe, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
                ("geo", cluster_simil, ["latitude", "longitude"]),
                ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
            ],
            remainder=default_num_pipeline)  # one column remaining: housing_median_age
        return preprocessing.fit_transform(X)

