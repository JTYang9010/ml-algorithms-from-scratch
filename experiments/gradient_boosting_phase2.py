import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from src.gradient_boosting import my_GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# syn regression dataset

rng_data = np.random.default_rng(42)

n_samples = 1000
n_features = 8 

X = rng_data.uniform(-1,1,size=(n_samples,n_features))

def true_function(X):
    return(np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1] ** 2 + X[:, 0] * X[:, 2])

noise = rng_data.normal(loc = 0, scale= 0.1, size= n_samples)

y = true_function(X) + noise

# data split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.15 / 0.85,
    random_state=42
)

learning_rates = [1, 0.3, 0.1, 0.03]

max_trees = 300
max_depth = 2

results = {}

for lr in learning_rates:
    print()
    print("=" * 60)
    print(
        f"Training learning_rate = "
        f"{lr}"
    )
    print("=" * 60)

    model = my_GradientBoostingRegressor(
        n_estimators=max_trees,
        lr=lr,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X_train,y_train)

    train_mse_history = []
    val_mse_history = []
    test_mse_history = []

    for pred in model.staged_predict(X_train):
        train_mse_history.append(mean_squared_error(y_train,pred))

    for pred in model.staged_predict(X_val):
        val_mse_history.append(mean_squared_error(y_val,pred))

    for pred in model.staged_predict(X_test):
        test_mse_history.append(mean_squared_error(y_test,pred))

    train_mse_history = np.array(train_mse_history)
    val_mse_history = np.array(val_mse_history)
    test_mse_history = np.array(test_mse_history)

    best_stage = (np.argmin(val_mse_history)+ 1)

    best_val_mse = (val_mse_history[best_stage - 1])

    corresponding_test_mse = (test_mse_history[best_stage - 1])

    print(
        f"Best stage: {best_stage}"
    )

    print(
        f"Best validation MSE: "
        f"{best_val_mse:.6f}"
    )

    print(
        f"Test MSE at best stage: "
        f"{corresponding_test_mse:.6f}"
    )

    results[lr] = {
        "model": model,
        "train_mse": train_mse_history,
        "val_mse": val_mse_history,
        "test_mse": test_mse_history,
        "best_stage": best_stage,
        "best_val_mse": best_val_mse,
        "test_mse_at_best": (
            corresponding_test_mse
        )
    }

# Val MSE vs Boosting round
boosting_rounds = np.arange(1,max_trees + 1)

plt.figure(figsize=(9, 6))

for lr in learning_rates:

    plt.plot(
        boosting_rounds,
        results[
            lr
        ]["val_mse"],
        label=(
            f"learning_rate="
            f"{lr}"
        )
    )

plt.xlabel("Boosting Round")
plt.ylabel("Validation MSE")
plt.title("Learning Rate vs Boosting Rounds")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



plt.figure(figsize=(9, 6))

for learning_rate in learning_rates:

    plt.plot(
        boosting_rounds,
        results[
            learning_rate
        ]["train_mse"],
        label=(
            f"learning_rate="
            f"{learning_rate}"
        )
    )

plt.xlabel("Boosting Round")
plt.ylabel("Training MSE")
plt.title("Training Convergence for Different Learning Rates")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Summary
print()
print("=" * 75)
print("Summary")
print("=" * 75)

print(
    f"{'Learning Rate':<16}"
    f"{'Best Stage':<15}"
    f"{'Best Val MSE':<18}"
    f"{'Test MSE':<15}"
)

for lr in learning_rates:

    result = results[
        lr
    ]

    print(
        f"{lr:<16}"
        f"{result['best_stage']:<15}"
        f"{result['best_val_mse']:<18.6f}"
        f"{result['test_mse_at_best']:<15.6f}"
    )