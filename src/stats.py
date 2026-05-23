import numpy as np
import pandas as pd

def calculate_imbalance_metrics(labels):
    """
    Calculate the Imbalance Ratio (IR) and Gini coefficient for a categorical distribution.
    
    Parameters:
    - labels: array-like, sequence of class labels (e.g., cell types).
    
    Returns:
    - dict: Contains 'IR' and 'Gini' metrics.
    """
    counts = pd.Series(labels).value_counts().values
    
    if len(counts) == 0:
        return {"IR": 0.0, "Gini": 0.0}
    
    # Imbalance Ratio: Majority class count / Minority class count
    min_count = counts.min()
    ir = counts.max() / min_count if min_count > 0 else np.inf
    
    # Gini Coefficient of inequality (0 = perfect balance, 1 = extreme imbalance)
    counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * counts).sum() / (n * counts.sum())
    
    return {
        "IR": ir,
        "Gini": gini,
        "Majority_Count": counts.max(),
        "Minority_Count": counts.min()
    }