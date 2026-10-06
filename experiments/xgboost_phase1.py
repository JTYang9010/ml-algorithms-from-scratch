import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ==========================================================
# Project path
# ==========================================================

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(project_root)


# ==========================================================
# Import XGBoost Tree
# ==========================================================

from src.xgboost_tree import XGBoostTree



# ==========================================================
# 1. Synthetic Regression Dataset
# ==========================================================

rng = np.random.default_rng(42)


n_samples = 500
n_features = 3


X = rng.uniform(
    -1,
    1,
    size=(
        n_samples,
        n_features
    )
)


def true_function(X):

    return (
        np.sin(
            2*np.pi*X[:,0]
        )
        +
        0.5*X[:,1]**2
        +
        X[:,0]*X[:,2]
    )


noise = rng.normal(
    0,
    0.1,
    size=n_samples
)


y = (
    true_function(X)
    +
    noise
)



# ==========================================================
# 2. Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ==========================================================
# 3. Initial Prediction
# ==========================================================
#
# XGBoost first tree:
#
# F0(x)=0
#
# gradient = prediction - y
#
# hessian = 1
#
# ==========================================================


initial_prediction = np.zeros_like(
    y_train
)


gradients = (
    initial_prediction
    -
    y_train
)


hessians = np.ones_like(
    y_train
)



print("="*70)

print("Gradient / Hessian Check")

print("="*70)


print(
    "Gradient mean:",
    np.mean(gradients)
)


print(
    "Hessian unique:",
    np.unique(hessians)
)



# ==========================================================
# 4. Train XGBoost Tree
# ==========================================================


tree = XGBoostTree(
    max_depth=3,
    min_samples_split=10,
    lambda_reg=1.0
)


tree.fit(
    X_train,
    gradients,
    hessians
)



print()

print("="*70)

print("Tree Training Finished")

print("="*70)



# ==========================================================
# 5. Prediction
# ==========================================================


y_pred = tree.predict(
    X_test
)



test_mse = mean_squared_error(
    y_test,
    y_pred
)


print(
    "Test MSE:",
    test_mse
)



# ==========================================================
# 6. Compare Prediction
# ==========================================================


plt.figure(
    figsize=(8,5)
)


# prediction scatter
plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    label="XGBoost Tree Prediction"
)


# y=x reference line

min_val = min(
    np.min(y_test),
    np.min(y_pred)
)

max_val = max(
    np.max(y_test),
    np.max(y_pred)
)


plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label=r"$y=x$"
)


plt.xlabel(
    "True y"
)


plt.ylabel(
    "Prediction"
)


plt.title(
    "XGBoost Tree Phase 1 Prediction"
)


plt.legend()


plt.grid(
    alpha=0.3
)
plt.tight_layout()

plt.show()


# ==========================================================
# 7. Inspect Root Split
# ==========================================================


root = tree.root


print()

print("="*70)

print("Root Node Information")

print("="*70)


if root.is_leaf:

    print(
        "Root is leaf"
    )

else:

    print(
        "Split feature:",
        root.feature_index
    )


    print(
        "Threshold:",
        root.threshold
    )



# ==========================================================
# 8. Check Leaf Weights
# ==========================================================


def print_leaf(
    node,
    depth=0
):

    if node.is_leaf:

        print(
            " "*depth,
            "Leaf weight:",
            node.prediction
        )

        return


    print_leaf(
        node.left,
        depth+1
    )


    print_leaf(
        node.right,
        depth+1
    )



print()

print("="*70)

print("Leaf Weights")

print("="*70)


print_leaf(
    tree.root
)