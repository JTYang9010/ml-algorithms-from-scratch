import os
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.decision_tree import DecisionTreeRegressorScratch
from src.bootstrap import bootstrap_sample

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

# Bootstrap Experiment

rng_bootstrap = np.random.default_rng(123)

n_trees = 5 

trees = []

unique_sample_counts = []
oob_sample_counts = []
oob_percentages = []
oob_mse_list = []

for tree_id in range(n_trees):
    (X_boot,y_boot,bootstrap_idx,oob_idx) = bootstrap_sample(X_train,y_train,rng_bootstrap)

    unique_bootstrap = np.unique(bootstrap_idx)

    unique_count = len(unique_bootstrap)
    oob_count = len(oob_idx)

    oob_percentage = (oob_count / len(X_train) * 100)

    unique_sample_counts.append(unique_count)

    oob_sample_counts.append(oob_count)

    oob_percentages.append(oob_percentage)

    print("\n" + "=" * 60)
    print(f"Tree {tree_id + 1}")
    print("=" * 60)

    print("First 30 bootstrap indices:")

    print(bootstrap_idx[:30])

    print(
        f"Bootstrap size: "
        f"{len(bootstrap_idx)}"
    )

    print(
        f"Unique samples: "
        f"{len(unique_bootstrap)}"
    )

    print(
        f"OOB samples: "
        f"{len(oob_idx)}"
    )

    print(
        f"OOB percentage: "
        f"{len(oob_idx) / len(X_train) * 100:.2f}%"
    )

    print("First 20 OOB indices:")

    print(oob_idx[:20])

    tree = DecisionTreeRegressorScratch(
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1
    )

    tree.fit(X_boot, y_boot)

    trees.append(tree)

    # Use OOB as evaluation

    if len(oob_idx) > 0:

        X_oob = X_train[oob_idx]
        y_oob = y_train[oob_idx]

        oob_pred = tree.predict(X_oob)

        oob_mse = mean_squared_error(y_oob,oob_pred)

        print(f"OOB MSE: {oob_mse:.4f}")

sample = X_test[0:1]

for i, tree in enumerate(trees):

    prediction = tree.predict(sample)[0]

    print(
        f"Tree {i + 1}: "
        f"{prediction:.4f}"
    )

# plot
import matplotlib.pyplot as plt

tree_ids = np.arange(1,n_trees + 1)

unique_sample_counts = np.array(unique_sample_counts)

oob_sample_counts = np.array(oob_sample_counts)

plt.figure(figsize=(8, 5))

plt.bar(
    tree_ids,
    unique_sample_counts,
    label="Unique Bootstrap Samples"
)

plt.bar(
    tree_ids,
    oob_sample_counts,
    bottom=unique_sample_counts,
    label="OOB Samples"
)

plt.xlabel("Tree")
plt.ylabel("Number of Original Training Samples")
plt.title("Bootstrap Composition for Each Tree")
plt.xticks(tree_ids)
plt.legend()
plt.grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.show()