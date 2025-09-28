from sklearn.base import BaseEstimator, TransformerMixin
from constants import NUMERIC_COLS, CATEGORICAL_COLS

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specified features from DataFrame
    if nothing is passed then default combinations of numeric and categorical features will be selected
    """
    def __init__(self, feature_names=None):
        # Allow passing feature names through constructor
        self.feature_names = feature_names if feature_names is not None else (NUMERIC_COLS + CATEGORICAL_COLS)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]

class CategoricalMarker(BaseEstimator, TransformerMixin):
    """Mark categorical columns for tree-based models"""
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features if categorical_features is not None else CATEGORICAL_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure categorical features are string type
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].astype('category')
        return X

class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Transform ordinal categorical columns to numeric labels"""

    # TODO: this can be used later if 2 columns (Designation and ProductPitched) doesnt work well on categorical nature

    # Define ordinal mappings
    DESIGNATION_MAP = {
        'Executive': 1,
        'Manager': 2,
        'Senior Manager': 3,
        'AVP': 4,
        'VP': 5
    }

    PRODUCT_MAP = {
        'Basic': 1,
        'Standard': 2,
        'Deluxe': 3,
        'Super Deluxe': 4,
        'King': 5
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Designation'] = X['Designation'].map(self.DESIGNATION_MAP)
        X['ProductPitched'] = X['ProductPitched'].map(self.PRODUCT_MAP)
        return X
