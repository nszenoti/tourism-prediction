import pandas as pd
# cross-folder imports
from src.utils.hf_utils import HFHandler
from src.utils.logger import get_logger
# sibling files
from tracking_utils import MLflowTracker
from pipeline_utils import CatBoostTrainer, LightGBMTrainer, save_pipeline
from eval_utils import select_best_model

logger = get_logger(__name__)

def load_data():
    """Load data from HuggingFace"""
    logger.info("Loading data from HuggingFace...")

    base_path = "hf://datasets/nszfgfg/tourism-prediction"

    # Load splits
    X_train = pd.read_csv(f"{base_path}/Xtrain.csv")
    X_test = pd.read_csv(f"{base_path}/Xtest.csv")
    y_train = pd.read_csv(f"{base_path}/ytrain.csv")
    y_test = pd.read_csv(f"{base_path}/ytest.csv")

    logger.info("Data loaded successfully")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, tracker):
    """Train and track multiple models"""
    logger.info("Starting model training...")

    catboost_base_params = {
        "auto_class_weights": "Balanced",  # handle imbalance
        "random_seed": 42,
        "verbose": False,
        "eval_metric": "AUC",  # or 'Accuracy', 'F1', etc.
        "loss_function": "Logloss", # Binary CLassification
    }
    lightgbm_base_params = {
        "random_state": 42,
        "verbose": -1,
        "metric": "auc",         # equivalent to CatBoost's "AUC"
        "objective": "binary",   # equivalent to CatBoost's "Logloss"
        "is_unbalance": True    # equivalent to CatBoost's "auto_class_weights"
    }
    catboost_tune_params = {
        "iterations": [300, 600],           # shorter training (instead of 1000+)
        "learning_rate": [0.05, 0.1],       # safe small range
        "depth": [4, 6],                    # shallow vs medium trees
        "l2_leaf_reg": [3, 5],              # regularization strength
        "subsample": [0.8],                 # fixed, but included for realism
        "colsample_bylevel": [0.8]          # same here
    }
    lightgbm_tune_params = {
        "n_estimators": [300, 600],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "reg_lambda": [3, 5],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    }

    # Train models
    models = []

    # CatBoost
    logger.info("Training CatBoost...")
    tag1 = 'CatBoost'
    trainer1 = CatBoostTrainer(base_params=catboost_base_params)
    catboost_pipe, catboost_meta = trainer1.train(
        X_train, y_train,
        tracker=tracker,
        tag=tag1
    )
    models.append((tag1, catboost_pipe, catboost_meta))

    # LightGBM
    logger.info("Training LightGBM...")
    tag2 = 'LightGBM'
    trainer2 = LightGBMTrainer(base_params=lightgbm_base_params)
    lightgbm_pipe, lightgbm_meta = trainer2.train(
        X_train, y_train,
        tracker=tracker,
        tag=tag2
    )
    models.append((tag2, lightgbm_pipe, lightgbm_meta))

    # Catboost with tuned params
    logger.info("Training CatBoost with tuned params...")
    tag3 = 'CatBoost Tuned'
    trainer3 = CatBoostTrainer(base_params=catboost_base_params, hyper_params=catboost_tune_params)
    catboost_tuned_pipe, catboost_tuned_meta = trainer3.train(
        X_train, y_train,
        tracker=tracker,
        tag=tag3
    )
    models.append((tag3, catboost_tuned_pipe, catboost_tuned_meta))

    # LightGBM with tuned params
    logger.info("Training LightGBM with tuned params...")
    tag4 = 'LightGBM Tuned'
    trainer4 = LightGBMTrainer(base_params=lightgbm_base_params, hyper_params=lightgbm_tune_params)
    lightgbm_tuned_pipe, lightgbm_tuned_meta = trainer4.train(
        X_train, y_train,
        tracker=tracker,
        tag=tag4
    )
    models.append((tag4, lightgbm_tuned_pipe, lightgbm_tuned_meta))

    return models

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Initialize MLflow tracker
    tracker = MLflowTracker(experiment_name="tourism-prediction")

    # Train models
    models = train_models(X_train, y_train, tracker)

    # Select best model
    logger.info("Selecting best model...")
    best_model = select_best_model(models, X_train, y_train, X_test, y_test)

    # Log best model to mlflow
    tracker.log_best_model(best_model, X_train, X_test, y_train, y_test)

    # Save best model
    model_path = save_pipeline(best_model['pipe'])
    logger.info(f"Best model saved: {best_model['name']}")

    # Upload to HuggingFace
    logger.info("Uploading model to HuggingFace...")
    hf = HFHandler()
    hf.upload(model_path, "tourism-model", repo_type="model")

    # Upload MLflow runs
    logger.info("Uploading MLflow tracking data...")
    # mlflow by default saves things in root dir ie /mlruns is dir name
    hf.upload("mlruns", "tourism-experiments", repo_type="model")

if __name__ == "__main__":
    main()
