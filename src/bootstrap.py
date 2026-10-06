import numpy as np

def bootstrap_sample(X,y,rng):

    # X:(n_samples,n_features)

    n_samples = X.shape[0]

    bootstrap_idx = rng.choice(n_samples,size = n_samples, replace = True) 

    X_boot = X[bootstrap_idx]
    y_boot = y[bootstrap_idx]

    selected_unique = np.unique(bootstrap_idx)

    oob_mask = np.ones(n_samples,dtype=bool)

    oob_mask[selected_unique] = False

    oob_idx = np.flatnonzero(oob_mask)

    return (X_boot,y_boot,bootstrap_idx,oob_idx)