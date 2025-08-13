import numpy as np

def rps(probs, outcome_index):
    """Ranked Probability Score for 3-class outcomes (H,D,A).
    probs: array shape (n,3); outcome_index: array of 0,1,2
    """
    K = 3
    o = np.eye(K)[outcome_index]
    cprobs = np.cumsum(probs, axis=1)
    coutcomes = np.cumsum(o, axis=1)
    return np.mean(np.sum((cprobs - coutcomes)**2, axis=1) / (K-1))
