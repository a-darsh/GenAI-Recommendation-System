import numpy as np

def map_score(y_true, y_pred):
    """
    Compute Mean Average Precision (MAP) score.
    Args:
        y_true (list of lists): True labels.
        y_pred (list of lists): Predicted labels.
    Returns:
        float: MAP score.
    """
    average_precisions = []
    for true, pred in zip(y_true, y_pred):
        if not true:
            continue
        hit = 0
        sum_precisions = 0
        for i, p in enumerate(pred):
            if p in true and p not in pred[:i]:
                hit += 1
                sum_precisions += hit / (i + 1.0)
        if hit == 0:
            average_precisions.append(0)
        else:
            average_precisions.append(sum_precisions / len(true))
    return np.mean(average_precisions)

def ndcg_score(y_true, y_pred, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) score.
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        k (int): Number of top scored items to consider.
    Returns:
        float: NDCG score.
    """
    def dcg(scores):
        return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(2, scores.size + 2))), dtype=np.float32)

    actual = np.zeros(len(y_pred))
    for i, val in enumerate(y_true):
        actual[i] = val in y_pred[:k]
    
    best = dcg(np.sort(actual)[::-1])
    actual_dcg = dcg(actual)
    return actual_dcg / best if best > 0 else 0

def rmse_score(y_true, y_pred):
    """
    Compute Root Mean Square Error (RMSE).
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
    Returns:
        float: RMSE score.
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

# Example usage
if __name__ == "__main__":
    y_true = [[1, 2, 3], [1, 2, 3, 4]]
    y_pred = [[1, 2, 4], [1, 2, 4, 5]]
    print("MAP Score:", map_score(y_true, y_pred))
    print("NDCG Score:", ndcg_score(y_true[0], y_pred[0], k=3))
    print("RMSE Score:", rmse_score(np.array([1, 2, 3]), np.array([1.1, 2.2, 3.3])))
