import numpy as np

class XGBNode:

    def __init__(
        self,
        prediction=None
    ):
        self.prediction = prediction
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True


class XGBoostTree:

    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        lambda_reg=1.0
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg
        self.root = None

    def calculate_leaf_weight(
        self,
        gradients,
        hessians
    ):
        G = np.sum(gradients)
        H = np.sum(hessians )
        weight = (-G /(H + self.lambda_reg))

        return weight

    def calculate_gain(
        self,
        g_left,
        h_left,
        g_right,
        h_right
    ):

        G_left = np.sum(g_left)
        H_left = np.sum(h_left)

        G_right = np.sum(g_right)
        H_right = np.sum(h_right)

        G_total = (G_left + G_right)
        H_total = (H_left + H_right)

        gain = 0.5 * (
            G_left**2 /(H_left + self.lambda_reg)
            + G_right**2 /(H_right+self.lambda_reg)
            - G_total**2 /(H_total+self.lambda_reg)
        )

        return gain

    def fit(
        self,
        X,
        gradients,
        hessians
    ):

        self.root = self._build_tree(
            X,
            gradients,
            hessians,
            depth=0
        )

    def best_split(
        self,
        X,
        gradients,
        hessians
    ):

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature in range(n_features):

            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:


                left_idx = (
                    X[:, feature]
                    <= threshold
                )

                right_idx = ~left_idx


                if (np.sum(left_idx) == 0
                    or
                    np.sum(right_idx) == 0
                ):
                    continue


                gain = self.calculate_gain(
                    gradients[left_idx],
                    hessians[left_idx],
                    gradients[right_idx],
                    hessians[right_idx]
                )


                if gain > best_gain:

                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold


        return (best_feature,best_threshold)

    def _build_tree(
        self,
        X,
        gradients,
        hessians,
        depth
    ):

        node = XGBNode()

        if (depth >= self.max_depth
            or 
            len(X) < self.min_samples_split
        ):

            node.prediction = (
                self.calculate_leaf_weight(
                    gradients,
                    hessians
                )
            )

            return node

        feature, threshold = (
            self.best_split(
                X,
                gradients,
                hessians
            )
        )

        if feature is None:
            node.prediction = (
                self.calculate_leaf_weight(
                    gradients,
                    hessians
                )
            )

            return node

        node.is_leaf=False
        node.feature_index = feature
        node.threshold = threshold

        left_idx = (X[:,feature]<= threshold)
        right_idx = ~left_idx

        node.left = self._build_tree(
            X[left_idx],
            gradients[left_idx],
            hessians[left_idx],
            depth+1
        )


        node.right = self._build_tree(
            X[right_idx],
            gradients[right_idx],
            hessians[right_idx],
            depth+1
        )

        return node


    def predict_one(
        self,
        x,
        node
    ):

        if node.is_leaf:
            return node.prediction

        if (x[node.feature_index]<= node.threshold):
            return self.predict_one(x,node.left )

        else:
            return self.predict_one(x,node.right)

    def predict(
        self,
        X
    ):
        predictions=[]
        for x in X:

            predictions.append(
                self.predict_one(
                    x,
                    self.root
                )
            )

        return np.array(predictions)