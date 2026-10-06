import numpy as np

from src.decision_tree_ver2 import DecisionTreeRegressorScratch

class my_GradientBoostingRegressor:

    def __init__(
        self,
        n_estimators = 100,
        lr = 0.1,
        max_depth = 3,
        min_samples_split =2,
        min_samples_leaf=1,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.trees = []

        self.init_pred = None

        self.train_mse_history = []
        self.mean_abs_residual_history = []

    def fit(self,X,y):

        X = np.asarray(X,dtype=float)
        y = np.asarray(y,dtype=float)

        if X.ndim != 2: raise ValueError("X must be 2D")
        if y.ndim != 1: raise ValueError("y must be 1D")
        if len(X) != len(y): raise ValueError("X and y must have same length")

        # State Reset
        self.trees = []
        self.train_mse_history = []
        self.mean_abs_residual_history = []

        rng = np.random.default_rng(self.random_state)

        # Init model

        self.init_pred = np.mean(y)
        curr_pred = np.full(shape=len(y), fill_value= self.init_pred, dtype= float)

        # Seq. Boosting

        for stage in range(self.n_estimators):

            residual = y-curr_pred

            tree_seed = rng.integers(0,np.iinfo(np.int32).max)

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                rand_state=int(tree_seed)
            )
                
            tree.fit(X,residual)

            correction = tree.predict(X)
            curr_pred += (self.lr * correction)

            self.trees.append(tree)

            mse = np.mean(y-curr_pred)**2

            new_residual = y - curr_pred

            mean_abs_residual = np.mean(np.abs(new_residual))

            self.train_mse_history.append(mse)
            self.mean_abs_residual_history.append(mean_abs_residual)
        
        return self

    def predict(self,X):

        if self.init_pred is None: raise ValueError("Model not fitted")

        X = np.asarray(X,dtype=float)
        prediction = np.full(shape = X.shape[0],fill_value= self.init_pred,dtype= float)

        for tree in self.trees:
            prediction += self.lr * tree.predict(X)

        return prediction

    def staged_predict(self, X): #for observation

        if self.init_pred is None:
            raise ValueError("Model not fitted.")

        X = np.asarray(X,dtype=float)

        prediction = np.full(shape=X.shape[0],fill_value=self.init_pred,dtype=float)

        for tree in self.trees:

            prediction += self.lr * tree.predict(X)

            yield prediction.copy()