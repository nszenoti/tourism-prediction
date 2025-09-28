from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                           precision_score, recall_score)
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def study_results(results, metric):
    """Helper function to analyze model results
    NOTE: This is only for local run for debug/dev purposes
    """
    # Create DataFrame for easy analysis
    results_df = pd.DataFrame([
        {
            'name': r['name'],
            'train_score': r['train_score'],
            'test_score': r['test_score'],
            'gap': r['gap'],
            'selection_score': r['selection_score']
        } for r in results
    ])
    print(f"\nModel Performance Summary ({metric}):")
    print(results_df)
    return results_df

def select_best_model(pipes, X_train, y_train, X_test, y_test, metric='roc_auc'):
    """
    Selects the best model from a list of pipelines based on a specified evaluation metric.

    Args:
        pipes: list of (name, pipeline, meta) tuples
        X_train, y_train: training data
        X_test, y_test: test data
        metric: evaluation metric to use ('accuracy', 'f1', 'roc_auc', 'precision', 'recall')

    Returns:
        dict: A dictionary containing information about the best model, including:
            - 'name': model name
            - 'pipe': the fitted pipeline/model
            - 'train_score': metric score on training data
            - 'test_score': metric score on test data
            - 'gap': generalization gap (train_score - test_score)
            - 'meta': any additional meta info
            - 'selection_score': score used for ranking models
    """
    METRIC_FUNCS = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score,
        'precision': precision_score,
        'recall': recall_score
    }

    if metric not in METRIC_FUNCS:
        raise ValueError(f"Metric {metric} not supported. Choose from {list(METRIC_FUNCS.keys())}")

    # ! NOTE: For now only tracking 1 score but can compute all scores and take decision based on one !!

    metric_func = METRIC_FUNCS[metric]
    results = []

    for name, pipe, meta in pipes:
        try:
            # Your existing prediction and scoring code
            y_train_pred = pipe.predict(X_train)
            y_test_pred = pipe.predict(X_test)

            if metric == 'roc_auc':
                # For classifiers that have predict_proba
                if hasattr(pipe, 'predict_proba'):
                    y_train_pred = pipe.predict_proba(X_train)[:,1]
                    y_test_pred = pipe.predict_proba(X_test)[:,1]

            train_score = metric_func(y_train, y_train_pred)
            test_score = metric_func(y_test, y_test_pred)
            generalization_gap = train_score - test_score
            selection_score = test_score - abs(generalization_gap)

            logger.info(f"EVAL || {name} :- perf > {selection_score} ||")

            results.append({
                'name': name,
                'pipe': pipe,
                'train_score': train_score,
                'test_score': test_score,
                'gap': generalization_gap,
                'meta': meta,
                'selection_score': selection_score
            })
        except Exception as e:
            logger.error(f"Error evaluating model {name}: {str(e)}")

    if not results:
        raise ValueError("No models were successfully evaluated")

    # ! Uncomment below for local run for debug/dev purposes
    # results_df = study_results(results, metric)

    results_sorted = sorted(results, key=lambda x: x['selection_score'], reverse=True)

    return results_sorted[0]
