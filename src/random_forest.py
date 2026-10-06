import numpy as np

from src.decision_tree_ver2 import DecisionTreeRegressorScratch
from src.bootstrap import bootstrap_sample

class RandomForestRegressorScratch:

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        rand_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rand_state = rand_state

        self.trees = []
        self.bootstrap_indices_ = []
        self.oob_indices_ = []

    def fit(self,X,y):

        X = np.asarray(X, dtype =float)
        y = np.asarray(y, dtype =float)

        if X.ndim !=2:
            raise ValueError("X is not a 2D arrray")

        if y.ndim !=1:
                    raise ValueError("y is not a 1D arrray")

        if len(X) != len(y):
              raise ValueError("X and y do mot match in length")

        self.trees = []
        self.bootstrap_indices_ = []
        self.oob_indices_ = []

        rng = np.random.default_rng(self.rand_state)

        for tree_id in range(self.n_estimators):

            # Bootstrap 
            (X_boot, y_boot,bootstrap_indices,oob_indices) = bootstrap_sample(X,y,rng)

            self.bootstrap_indices_.append(bootstrap_indices)
            self.oob_indices_.append(oob_indices)

            #random seed generator
            tree_seed = rng.integers(0,np.iinfo(np.int32).max) # 32 位元有號整數最大值

            # random CART tree
            tree = DecisionTreeRegressorScratch(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            rand_state=int(tree_seed)
            )

            # bootstrap train

            tree.fit(X_boot,y_boot)

            self.trees.append(tree)

        return self

    def predict(self, X):
         if len(self.trees) == 0:
              raise ValueError("Empty forest")

         X = np.asarray(X,dtype=float)

         tree_predictions = np.array([tree.predict(X) for tree in self.trees])

         # Aggregation(Average)
         forest_prediction = np.mean(tree_predictions,axis=0) #Average along tree axis

         return forest_prediction

    def predict_all_trees(self,X):

         if len(self.trees) == 0:
              raise ValueError( "Empty forest")

         X = np.asarray(X,dtype=float)

         return np.array([tree.predict(X) for tree in self.trees])