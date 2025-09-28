import mlflow
from sklearn.metrics import classification_report, roc_auc_score

class MLflowTracker:
    """Helper class for MLflow experiment tracking"""

    def __init__(self, experiment_name="tourism-prediction"):
        mlflow.set_experiment(experiment_name)

    def log_grid_search_results(self, grid_search, model_name):
        """Log each grid search CV result as separate run"""
        results = grid_search.cv_results_

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)

            for idx in range(len(results['params'])):
                with mlflow.start_run(nested=True):
                    mlflow.log_params(results['params'][idx])
                    mlflow.log_metrics({
                        "mean_test_score": results["mean_test_score"][idx],
                        "std_test_score": results["std_test_score"][idx],
                        "rank_test_score": results["rank_test_score"][idx]
                    })

    def log_best_model(self, model_info, X_train, X_test, y_train, y_test):
        """Log best model with comprehensive metrics"""
        with mlflow.start_run(run_name="best_model"):
            # Log model info
            mlflow.log_params({
                "model_name": model_info['name'],
                **model_info['meta']
            })

            # Get predictions
            pipe = model_info['pipe']
            y_train_pred = pipe.predict(X_train)
            y_test_pred = pipe.predict(X_test)
            y_train_proba = pipe.predict_proba(X_train)[:, 1]
            y_test_proba = pipe.predict_proba(X_test)[:, 1]

            # Compute detailed metrics
            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            test_report = classification_report(y_test, y_test_pred, output_dict=True)

            # Log all metrics
            mlflow.log_metrics({
                # Original metrics
                "train_score": model_info['train_score'],
                "test_score": model_info['test_score'],
                "generalization_gap": model_info['gap'],

                # Additional detailed metrics
                "train_roc_auc": roc_auc_score(y_train, y_train_proba),
                "test_roc_auc": roc_auc_score(y_test, y_test_proba),
                "train_accuracy": train_report['accuracy'],
                "train_precision": train_report['1']['precision'],
                "train_recall": train_report['1']['recall'],
                "train_f1": train_report['1']['f1-score'],
                "test_accuracy": test_report['accuracy'],
                "test_precision": test_report['1']['precision'],
                "test_recall": test_report['1']['recall'],
                "test_f1": test_report['1']['f1-score']
            })

            # ! no need to log now as HF we are tracking seprately !!
            # Log the model
            # mlflow.sklearn.log_model(pipe, "model")

class ExperimentTracker:
    """Simple, clean MLflow experiment tracking"""

    def __init__(self, experiment_name="tourism-prediction"):
        # Set experiment and ensure local tracking
        # mlflow.set_tracking_uri('mlruns')
        mlflow.set_experiment(experiment_name)

    def log_model_results(self, model_name, params, metrics, model=None):
        """Single method to log model training results"""
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Optionally log model if provided
            if model:
                mlflow.sklearn.log_model(model, "model")

    def log_best_model(self, model_info):
        """Log best performing model with clear tags"""
        with mlflow.start_run(run_name="best_model"):
            mlflow.set_tag("status", "best_model")

            # Log model details
            mlflow.log_params({
                "model_type": model_info['name'],
                "selection_criteria": "test_score_with_generalization"
            })

            # Log performance metrics
            mlflow.log_metrics({
                "train_score": model_info['train_score'],
                "test_score": model_info['test_score'],
                "generalization_gap": model_info['gap']
            })

            # Log the model itself
            mlflow.sklearn.log_model(model_info['pipe'], "model")
