import os
import sys

# # Add current directory to path
# # so that we can directly used sibbling files
# sys.path.append(os.path.dirname(__file__))

import joblib
import time
from functools import wraps
from sklearn.pipeline import Pipeline
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# These work fine as they're in same directory (ie same dir as of running scripts at moment)
from custom_transformers import FeatureSelector, CategoricalMarker
from constants import CATEGORICAL_COLS

from src.utils.logger import get_logger

logger = get_logger(__name__)

def time_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        train_time = time.time() - start_time

        if isinstance(result, tuple) and isinstance(result[1], dict):
            model, meta = result
            meta['training_time'] = train_time
            return model, meta
        return result
    return wrapper

def get_pipeline_params(params, step_name='model'):
    """
    Convert regular params dict to pipeline format with step prefix
    Example: {'n_estimators': 100} -> {'model__n_estimators': 100}
    """
    return {f'{step_name}__{key}': value for key, value in params.items()}

def save_pipeline(pipeline):
    """Save trained pipeline with performance metrics & returns path to serialized file"""
    filePath = f"artifacts/selected_model.joblib"
    logger.info(f"Saving pipeline to {filePath}")

    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(pipeline, filePath)

    return filePath

class ModelTrainer(ABC):
    MODEL_STEP_NAME = 'model'

    def __init__(self, base_params, hyper_params=None):
        self.base_params = base_params
        self.hyper_params = hyper_params

    @abstractmethod
    def get_fit_params(self) -> dict:
        """Returns model specific fit parameters"""
        pass

    @time_me
    def train(self, X_train, y_train, tracker=None, tag='N.A') ->  tuple:
        """Train model and return (model, meta)"""
        pipe = self.get_pipeline()
        fit_params = get_pipeline_params(self.get_fit_params(), self.MODEL_STEP_NAME)
        trained_pipe = None

        # class_name = self.__class__.__name__

        logger.info(f"Starting training with {tag}")

        if self.hyper_params:
            logger.info("Performing hyperparameter tuning...")
            hyper_params_grid = get_pipeline_params(self.hyper_params, self.MODEL_STEP_NAME)
            # Do hyperparameter tuning
            tuner = GridSearchCV(
                pipe,
                hyper_params_grid,
                scoring="roc_auc", # this can be changed to any other metric
                cv=3, # since main goal is mlops, so to do things faster using less folds
                n_jobs=-1  # parallelize across grid search folds & configs
            )
            tuner.fit(X_train, y_train, **fit_params)
            trained_pipe = tuner.best_estimator_

            # Log grid search results if tracker provided
            if tracker:
                tracker.log_grid_search_results(tuner, tag)
        else:
            # Basic training
            pipe.fit(X_train, y_train, **fit_params)
            trained_pipe = pipe

        model = trained_pipe.named_steps[self.MODEL_STEP_NAME]
        params_used = model.get_params()

        meta = {
            'params': params_used,
        }

        # Log after we get the result
        logger.info(f"Training completed. Model performance: {model.best_score_}")
        # logger.info(f"Pipeline steps: {[step[0] for step in trained_pipe.steps]}")

        return trained_pipe, meta

    @abstractmethod
    def get_pipeline(self) -> Pipeline:
        """Returns pipeline with column selection and model"""
        pass

class CatBoostTrainer(ModelTrainer):
    def get_fit_params(self) -> dict:
        return {'cat_features': CATEGORICAL_COLS}

    def get_pipeline(self):
        """Returns pipeline with column selection and CatBoost model"""

        steps = [
            ('column_selector', FeatureSelector()),
            (self.MODEL_STEP_NAME, CatBoostClassifier(**self.base_params)) # Base Classifier
        ]
        return Pipeline(steps)

class LightGBMTrainer(ModelTrainer):
    def get_fit_params(self) -> dict:
        return {'categorical_feature': CATEGORICAL_COLS}

    def get_pipeline(self):
        """Returns pipeline with column selection and LightGBM model"""

        steps = [
            ('column_selector', FeatureSelector()),
            ('categorical_transformer', CategoricalMarker(CATEGORICAL_COLS)),
            (self.MODEL_STEP_NAME, LGBMClassifier(**self.base_params))  # Classifier instance, not Trainer
        ]
        return Pipeline(steps)
