import numpy as np
import math

def compute_monitoring_metrics(system_type, y_true, y_pred):
    y_true = np.array(y_true, ndmin=1)
    y_pred = np.array(y_pred, ndmin=1)
    
    metrics = []

    if system_type == "classification":
        n = len(y_true)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = [
            ("accuracy", accuracy),
            ("f1", f1),
            ("precision", precision),
            ("recall", recall)
        ]

    elif system_type == "regression":
        diff = y_true - y_pred
        n = len(y_true)
        
        mae = np.sum(np.abs(diff)) / n
        rmse = np.sqrt(np.sum(diff ** 2) / n)
        
        metrics = [
            ("mae", mae),
            ("rmse", rmse)
        ]

    elif system_type == "ranking":
        sort_indices = np.argsort(y_pred)[::-1]
        sorted_labels = y_true[sort_indices]
        
        k = 3
        top_k = sorted_labels[:k]
        relevant_in_top_k = np.sum(top_k)
        total_relevant = np.sum(y_true)
        
        p_at_3 = relevant_in_top_k / k
        r_at_3 = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
        
        metrics = [
            ("precision_at_3", p_at_3),
            ("recall_at_3", r_at_3)
        ]
    
    return metrics

