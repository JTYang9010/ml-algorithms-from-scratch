import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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

from src.random_forest import RandomForestRegressorScratch

forest = RandomForestRegressorScratch(
    n_estimators=5,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    rand_state=42
)

forest.fit(X_train,y_train)

forest_prediction = forest.predict(X_val)

all_tree_predictions = (forest.predict_all_trees(X_val))

sample_id = 0

print("\nPredictions for validation sample 0:")

for tree_id in range(forest.n_estimators):
    prediction = (all_tree_predictions[tree_id,sample_id])
    print(f"Tree {tree_id + 1}: "f"{prediction:.4f}")

print("\nForest prediction:")
print(f"{forest_prediction[sample_id]:.4f}")
print("True value:")
print(f"{y_val[sample_id]:.4f}")

#1、10、50、100 棵 Trees

tree_counts = [1,10,50,100]

scratch_train_mse = []
scratch_val_mse = []
scratch_test_mse = []

for n_trees in tree_counts:
    print(f"\nTraining scratch forest: "f"{n_trees} trees")

    forest = RandomForestRegressorScratch(
        n_estimators=n_trees,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        rand_state=42
    )

    forest.fit( X_train, y_train)

    train_pred = forest.predict(X_train)
    val_pred = forest.predict(X_val)
    test_pred = forest.predict(X_test)

    train_mse = mean_squared_error(y_train,train_pred)
    val_mse = mean_squared_error(y_val,val_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    scratch_train_mse.append(train_mse)
    scratch_val_mse.append(val_mse)
    scratch_test_mse.append(test_mse)
    print(f"Train MSE = {train_mse:.4f}")

    print(f"Val MSE   = {val_mse:.4f}")
    print(f"Test MSE  = {test_mse:.4f}")

#sklearn Sanity Check

sklearn_train_mse = []
sklearn_val_mse = []
sklearn_test_mse = []

for n_trees in tree_counts:
    model = RandomForestRegressor(
    n_estimators=n_trees,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
    )

    model.fit(X_train,y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val )
    test_pred = model.predict(X_test)

    sklearn_train_mse.append(mean_squared_error(y_train,train_pred))
    sklearn_val_mse.append(mean_squared_error(y_val,val_pred))
    sklearn_test_mse.append(mean_squared_error(y_test,test_pred))

#plot

plt.figure(figsize=(8, 5))

plt.plot(
    tree_counts,
    scratch_train_mse,
    marker="o",
    label="Train MSE"
)

plt.plot(
    tree_counts,
    scratch_val_mse,
    marker="o",
    label="Validation MSE"
)

plt.plot(
    tree_counts,
    scratch_test_mse,
    marker="o",
    label="Test MSE"
)

plt.xlabel("Number of Trees")
plt.ylabel("MSE")
plt.title("Scratch Random Forest: Number of Trees vs MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#plot

plt.figure(figsize=(8, 5))

plt.plot(
    tree_counts,
    scratch_val_mse,
    marker="o",
    label="Scratch RF"
)

plt.plot(
    tree_counts,
    sklearn_val_mse,
    marker="o",
    label="sklearn RF"
)

plt.xlabel("Number of Trees")
plt.ylabel("Validation MSE")
plt.title("Scratch vs sklearn Random Forest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()