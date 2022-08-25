import os
import pickle
from collections import Counter
from itertools import combinations
from typing import List
from xmlrpc.client import boolean

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # data processing,
import seaborn as sns
import sklearn.pipeline
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from seaborn import load_dataset
from sklearn import metrics, preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             get_scorer)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def drop_rows(
    input_data: pd.DataFrame, features: list, drop_val: str = " ?"
) -> pd.DataFrame:
    input_data.replace(drop_val, float("NaN"), inplace=True)
    for feature in features:
        input_data.dropna(subset=[feature], inplace=True)
    return input_data


class Drop_Columns(BaseEstimator, TransformerMixin):
    """
    Transformer which drops rows containing missing values (indicated with " ?").

    Attributes
    ----------
    columns:List[str]
        Column names of the columns that are taken into account.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


class Predict_Missing(BaseEstimator, TransformerMixin):

    """
    Transformer which predicts missing values (indicated with " ?").

    Attributes
    ----------
    pred_columns:List[str]
        Column names of the columns that are taken into account.
    print_cross_val_score:bool
        True if a crossvalidation with cv=5 should be performed during training.
    label_missing:bool
        True if a column indicating with the value was missing should be added.
    """

    def __init__(
        self, pred_columns: List[str], print_cross_val_score: bool, label_missing: bool
    ):
        self.label_missing = label_missing
        self.pred_columns = pred_columns
        self.print_cross_val_score = print_cross_val_score
        self.pipe = {}
        self.missing_labels = {}
        self.labelencoder = {}

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        data = X
        # Drop all rows conating ' ?' for training the model
        for pred_column in self.pred_columns:
            data = data.loc[data[pred_column] != " ?"]
        # Train model for each column containing missing data
        for pred_column in self.pred_columns:
            y_data = data[pred_column].copy()
            self.labelencoder[pred_column] = preprocessing.LabelEncoder()
            y_data = self.labelencoder[pred_column].fit_transform(y_data)
            X_data = data.drop(columns=[pred_column]).copy()
            # Define model
            model = lgb.LGBMClassifier(random_state=0)
            # Define categorical features
            cat_features = X_data.select_dtypes(include="object").columns.tolist()
            cat_features.sort()
            # OneHotencode catecorical data
            transformer = make_column_transformer(
                (OneHotEncoder(handle_unknown="ignore", sparse=False), cat_features),
                remainder="passthrough",
            )
            self.pipe[pred_column] = make_pipeline(transformer, model)
            self.pipe[pred_column].fit(X_data, y_data)
            # Print validation score on training set if print_cross_val_score was set to true
            if self.print_cross_val_score:
                pred_missing_accurac = cross_val_score(
                    self.pipe[pred_column], X_data, y_data, cv=5, scoring="accuracy"
                ).mean()
                print(
                    f"Predict missing in {pred_column}: Cross-val accuracy on training data: ",
                    pred_missing_accurac,
                )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # Predict missing values
        for pred_column in self.pred_columns:
            data = X.loc[X[pred_column] == " ?"]
            X_data = data.drop(columns=[pred_column])
            y_data = self.pipe[pred_column].predict(X_data)
            y_data = self.labelencoder[pred_column].inverse_transform(y_data)

            # Label rows with missing values
            if self.label_missing:
                labels = X[[pred_column]].copy()
                labels.loc[labels[pred_column] == " ?", pred_column] = 1
                labels.loc[labels[pred_column] != 1, pred_column] = 0
                self.missing_labels[pred_column] = labels
                indices = X.loc[X[pred_column] == " ?"].index

            # Asign predicted values to feature matrix
            X.loc[X[pred_column] == " ?", pred_column] = y_data

        # Asign missing label to feature matrix
        if self.label_missing:
            for pred_column in self.pred_columns:
                X[pred_column + "_missing"] = self.missing_labels[pred_column]
        return X


class Impute_Missing(BaseEstimator):
    """
    Transformer which imputes missing values (e.g. indicated with " ?").

    Attributes
    ----------
    missing_value:any
        Value to be imputed.
    pred_columns:List[str]
        Column names of the columns that are taken into account.
    strategy:str
        The imputation strategy.
        If "most_frequent", then replace missing using the most frequent value along each column.
    label_missing:bool
        True if a column indicating with the value was missing should be added.
    """

    def __init__(
        self,
        missing_value: any,
        pred_columns: List[str],
        strategy: str,
        label_missing: bool,
    ):
        self.missing_value = missing_value
        self.label_missing = label_missing
        self.pred_columns = pred_columns
        self.strategy = strategy
        self.missing_labels = {}
        self.fill_value = {}

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for pred_column in self.pred_columns:
            self.fill_value[pred_column] = X[pred_column].value_counts().idxmax()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for pred_column in self.pred_columns:
            # Label rows with missing values
            if self.label_missing:
                labels = X[[pred_column]].copy()
                labels.loc[labels[pred_column] == self.missing_value, pred_column] = 1
                labels.loc[labels[pred_column] != 1, pred_column] = 0
                X[pred_column + "_missing"] = labels

            # Fill missing values
            X.loc[X[pred_column] == self.missing_value, pred_column] = self.fill_value[
                pred_column
            ]
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self = self.fit(X)
        X = self.transform(X)
        return X


class InteractionsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_cache=False):
        self.use_cache = use_cache
        if self.use_cache:
            self.gbm = pickle.load(open("temp/interactions_model", "rb"))
            self.sorted_idx = pickle.load(open("temp/interactions_sorted_idx", "rb"))
        else:
            self.gbm = GradientBoostingRegressor(
                n_estimators=32, max_depth=4, random_state=0
            )
            self.sorted_idx = None

    def fit(self, X, y=None):
        if self.use_cache:
            pass
        else:
            X = pd.DataFrame(X)
            self.gbm.fit(X, y)
            result = permutation_importance(
                self.gbm, X, y, n_repeats=5, random_state=42, n_jobs=-1
            )
            self.sorted_idx = result.importances_mean.argsort()
            os.makedirs("temp", exist_ok=True)
            pickle.dump(self.gbm, open("temp/interactions_model", "wb"))
            pickle.dump(self.sorted_idx, open("temp/interactions_sorted_idx", "wb"))

    def transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X)
        interactions = {}
        for comb in list(combinations(X.columns[self.sorted_idx[-15:]], 2)):
            interactions[str(comb[0]) + "_x_" + str(comb[1])] = X[comb[0]] * X[comb[1]]
        new_df = pd.DataFrame(interactions)
        X = pd.concat([X, new_df], axis=1)
        X.columns = X.columns.astype(str)
        return X

    def fit_transform(self, X, y=None):
        X = pd.DataFrame(X)
        self.fit(X, y)
        out = self.transform(X)
        return out


class OrderFeatures(BaseEstimator, TransformerMixin):
    """Order features of X, according to a given list with the feature names.

    Args:
        orderedfeaturenames:List[str]
            List with column names in the requested  order.
    """

    def __init__(
        self,
        orderedfeaturesnames: List[str] = [
            "marital",
            "educational num",
            "education",
            "capital gain",
            "age",
            "occupation",
            "capital loss",
            "hours per week",
            "fnlwgt",
            "relationship",
            "workclass",
            "gender",
            "country",
            "race",
        ],
    ):
        self.orderedfeaturesnames = orderedfeaturesnames

    def fit(self, X, y=None):
        return self

    def transform(self, X=None) -> pd.DataFrame:
        X = X[self.orderedfeaturesnames]
        return X
