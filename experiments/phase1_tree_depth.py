import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# 讓 Python 找到 src/
project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(project_root)


from src.decision_tree import DecisionTreeRegressorScratch


# ==================================================
# 1. Synthetic regression dataset
# ==================================================

rng = np.random.default_rng(42) #seed

n_samples = 1000
n_features = 8

# training data x

X = rng.uniform(
    -1,
    1,
    size=(n_samples, n_features)
)


def true_function(X):
    return (
        np.sin(2 * np.pi * X[:, 0])
        + 0.5 * X[:, 1] ** 2
        + X[:, 0] * X[:, 2]
    )


noise = rng.normal(
    loc=0,
    scale=0.1,
    size=n_samples
)

y = true_function(X) + noise    # training data y


# ==================================================
# 2. Train / Validation / Test split (分兩次切割)
# ==================================================
# 70% train + 15% validation +15% test = 85% raw +15% test

X_train_val, X_test, y_train_val, y_test = train_test_split( # test,val 還沒分開
    X,
    y,
    test_size=0.15,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.15 / 0.85, # 85% raw 中取0.15 / 0.85 才會有 15% 當validation
    random_state=42
)


print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==================================================
# 3. Test different max_depth
# ==================================================

depths = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12
]

train_mse_list = []
val_mse_list = []
test_mse_list = []


for depth in depths:

    tree = DecisionTreeRegressorScratch(
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1
    )

    tree.fit(
        X_train,
        y_train
    )

    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)
    test_pred = tree.predict(X_test)

    train_mse = mean_squared_error(
        y_train,
        train_pred
    )

    val_mse = mean_squared_error(
        y_val,
        val_pred
    )

    test_mse = mean_squared_error(
        y_test,
        test_pred
    )

    train_mse_list.append(train_mse)
    val_mse_list.append(val_mse)
    test_mse_list.append(test_mse)

    print(
        f"depth={depth:2d} | "
        f"train MSE={train_mse:.4f} | "
        f"val MSE={val_mse:.4f} | "
        f"test MSE={test_mse:.4f}"
    )


# ==================================================
# 4. Plot
# ==================================================

plt.figure()

plt.plot(
    depths,
    train_mse_list,
    marker="o",
    label="Train MSE"
)

plt.plot(
    depths,
    val_mse_list,
    marker="o",
    label="Validation MSE"
)

plt.plot(
    depths,
    test_mse_list,
    marker="o",
    label="Test MSE"
)

plt.xlabel("Max Depth")
plt.ylabel("MSE")

plt.title(
    "CART Regression Tree: Depth vs MSE"
)

plt.legend()
plt.grid()

plt.show()