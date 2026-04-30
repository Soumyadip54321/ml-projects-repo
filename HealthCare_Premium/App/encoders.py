'''
Script that stores all the preprocessing functions.
'''

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin


# setup OOF-CV target encoder in a way that it can be serialized in a pipeline
class oof_cv_target_encoder(BaseEstimator, TransformerMixin):
    '''
    Class that perform leakage-free OOF-CV target encoding.
    '''

    def __init__(self, col, n_splits=5):
        self.col = col
        self.global_mean = 0
        self.mapping_ = 0
        self.n_splits = n_splits

    def fit(self, X, y):
        '''
        Function that updates global mean and mapping to help encode nseen/inference data.
        '''
        self.global_mean = y.mean()

        # create dataframe containing all features inclusive of target
        df = X.copy()
        df[y.name] = y

        # store mapping basis train set
        self.mapping_ = df.groupby([self.col])[y.name].agg('mean')

        return self

    def fit_transform(self, X, y):
        '''
        Function that performs OOF-CV target encoding of train set and returns it.
        This is executed during model training.
        '''
        # perform 5-fold CV
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # store target encoded output for chosen column
        X_out = X.copy()
        X = X.copy()

        for tr_idx, val_idx in kf.split(X):
            # split dataset into train and validation folds using positions
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val = X.iloc[val_idx]

            # map training features to their labels
            X_trfold = X_tr.join(y_tr)

            # compute target encoding across different values of col
            mapping = X_trfold.groupby([self.col])[y.name].agg('mean')

            # target encode col
            X_out.iloc[val_idx, X_out.columns.get_loc(self.col)] = X_val[self.col].map(mapping).fillna(y_tr.mean())

        # change datatype of target encoded column to numeric
        X_out[self.col] = pd.to_numeric(X_out[self.col], errors='coerce')

        # fetch updated global mean & mapping post OOF-CV encoding
        self.fit(X, y)

        return X_out

    def transform(self, X):
        '''
        function that executes during inference to target encode unseen data prior to feeding it as input to model.
        '''
        X = X.copy()
        X[self.col] = X[self.col].map(self.mapping_).fillna(self.global_mean)

        # ensure target encoded column is numeric
        X[self.col] = pd.to_numeric(X[self.col], errors='coerce')

        return X