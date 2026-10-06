import numpy as np

class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorScratch:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root = None

    def fit(self, X, y):  #維度符合->建樹
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        self.root = self._grow_tree(X, y, depth=0)

        return self

    def _grow_tree(self, X, y, depth):
        n_samples = X.shape[0]

        # -------------------------
        # Stopping conditions
        # -------------------------

        if self.max_depth is not None and depth >= self.max_depth:
            return self._create_leaf(y)

        if n_samples < self.min_samples_split:
            return self._create_leaf(y)

        if np.all(y == y[0]):
            return self._create_leaf(y)

        # -------------------------
        # Find best split
        # -------------------------

        feature_index, threshold = self._best_split(X, y)

        # 找不到合法 split
        if feature_index is None:
            return self._create_leaf(y)

        # -------------------------
        # Split data(Boolean Masking)
        # -------------------------

        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        # -------------------------
        # Recursive construction
        # -------------------------

        left_child = self._grow_tree(
            X_left,
            y_left,
            depth + 1
        )

        right_child = self._grow_tree(
            X_right,
            y_right,
            depth + 1
        )

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def _best_split(self, X, y):
        n_samples, n_features = X.shape

        best_loss = np.inf
        best_feature = None
        best_threshold = None

        # -------------------------
        # Try every feature j
        # -------------------------

        for feature_index in range(n_features):

            feature_values = X[:, feature_index]

            unique_values = np.unique(feature_values) #不重複數值排序(小到大)

            if len(unique_values) <= 1:
                continue

            # threshold 為相鄰數值中間值
            thresholds = (
                unique_values[:-1]
                + unique_values[1:]
            ) / 2.0

            # -------------------------
            # Try every threshold s
            # -------------------------

            for threshold in thresholds:

                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                # min_samples_leaf 限制:不希望樹枝分太細
                if n_left < self.min_samples_leaf:
                    continue

                if n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                loss = (
                    self._squared_error(y_left)
                    + self._squared_error(y_right)
                )

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _squared_error(self, y):
        mean_y = np.mean(y)

        return np.sum(
            (y - mean_y) ** 2
        )

    def _create_leaf(self, y):
        prediction = np.mean(y)

        return Node(value=prediction)

    def predict(self, X): #多筆資料整合
        X = np.asarray(X, dtype=float)

        predictions = np.array([
            self._predict_one(sample, self.root)
            for sample in X
        ])

        return predictions

    def _predict_one(self, sample, node): #單筆資料在Decision Tree中移動

        if node.is_leaf():
            return node.value

        if sample[node.feature_index] <= node.threshold:
            return self._predict_one(
                sample,
                node.left
            )

        return self._predict_one(
            sample,
            node.right
        )

    def print_tree(self):
        self._print_node(self.root, depth=0)

    def _print_node(self, node, depth):
        indent = "    " * depth

        if node.is_leaf():
            print(
                f"{indent}Leaf: prediction = {node.value:.4f}"
            )
            return

        print(
            f"{indent}"
            f"X[{node.feature_index}] <= {node.threshold:.4f}"
        )

        print(f"{indent}Left:")
        self._print_node(
            node.left,
            depth + 1
        )

        print(f"{indent}Right:")
        self._print_node(
            node.right,
            depth + 1
        )