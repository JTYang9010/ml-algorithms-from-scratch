# How Random Feature Selection affects Training of Decision Tree ?
import os
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.decision_tree_ver2 import DecisionTreeRegressorScratch

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

# dataset split

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

# diagnostic tool
def tree_signature(node,depth = 0, max_depth = 2):
    if node.is_leaf():
        return("leaf",round(node.value,3))
    if depth >= max_depth:
        return(f"X{node.feature_index}",round(node.threshold,3))
    return(
        (f"X{node.feature_index}",round(node.threshold,3)),
        tree_signature(node.left,depth + 1,max_depth),
        tree_signature(node.right,depth + 1,max_depth)
    )

# 控制組: all features with 2 differrent seeds
tree_all_1 = DecisionTreeRegressorScratch(max_depth= 5,max_features=None,rand_state=1)
tree_all_2 = DecisionTreeRegressorScratch(max_depth= 5,max_features=None,rand_state=999)

tree_all_1.fit(X_train,y_train)
tree_all_2.fit(X_train,y_train)

# Comparision

sig_all_1 = tree_signature(tree_all_1.root)
sig_all_2 = tree_signature(tree_all_2.root)

print("\nALL FEATURES")
print("Tree seed=1:")
print(sig_all_1)
print("\nTree seed=999:")
print(sig_all_2)
print("\nSame structure:",sig_all_1 == sig_all_2)

# 實驗組:sqrt
# Same data + Same hyperparams + differrent seeds = differrent trees
tree_sqrt_1 = DecisionTreeRegressorScratch(max_depth= 5,max_features="sqrt",rand_state=1)
tree_sqrt_2 = DecisionTreeRegressorScratch(max_depth= 5,max_features="sqrt",rand_state=2)
tree_sqrt_3 = DecisionTreeRegressorScratch(max_depth= 5,max_features="sqrt",rand_state=3)

trees_sqrt = [tree_sqrt_1,tree_sqrt_2,tree_sqrt_3]
for tree in trees_sqrt:
    tree.fit(X_train,y_train)

for i, tree in enumerate(trees_sqrt, start=1):
    print(f"\nSeed {i}")
    print(tree_signature(tree.root))

# Prediction Comparision
probe = X_val[:5]
print("\nPredictions on same validation samples")

for seed, tree in zip([1, 2, 3],trees_sqrt):

    predictions = tree.predict(probe)

    print(f"Seed {seed}:",np.round(predictions,4))

# validation MSE Comparison
# MSE:  all feature selected < random feature selected
print("\nValidation MSE(sqrt)")

for seed, tree in zip([1, 2, 3],trees_sqrt):
    pred = tree.predict(X_val)
    mse = mean_squared_error(y_val,pred)
    print(f"Seed {seed}: "f"{mse:.4f}")

print("\nValidation MSE(all)")

pred_all_1 = tree_all_1.predict(X_val)
mse_all_1 = mean_squared_error(y_val,pred_all_1)
print(f"Seed {1}: "f"{mse_all_1:.4f}")