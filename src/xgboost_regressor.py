import numpy as np

from src.xgboost_tree import XGBoostTree


class MyXGBRegressor:

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        lambda_reg=1.0
    ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg

        self.trees = []



    def fit(
        self,
        X,
        y
    ):


        n_samples = len(y)

        # initial prediction

        prediction = np.zeros(
            n_samples
        )

        for m in range(
            self.n_estimators
        ):

            # gradient

            gradients = (prediction- y)


            # hessian

            hessians = np.ones(n_samples)


            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                lambda_reg=self.lambda_reg
            )


            tree.fit(
                X,
                gradients,
                hessians
            )


            update = tree.predict(X)


            prediction += (self.learning_rate * update)


            self.trees.append(tree)



    def predict(
        self,
        X
    ):
        prediction = np.zeros(len(X))

        for tree in self.trees:

            prediction += (self.learning_rate * tree.predict(X))

        return prediction