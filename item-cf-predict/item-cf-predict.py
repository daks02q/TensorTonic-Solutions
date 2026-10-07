import numpy as np 

def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    # Write code here
    ratings = np.asarray(user_ratings, dtype = float)
    item_sim = np.asarray(item_similarities, dtype = float)
    not_target = np.arange(len(ratings)) != target
    mask = (item_sim > 0) & (ratings > 0) & not_target
    calc_preds = np.sum((ratings[mask] * item_sim[mask])) / np.sum(item_sim[mask])
    return calc_preds