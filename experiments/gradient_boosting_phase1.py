import os
import sys

import numpy as np
import matplotlib.pyplot as plt

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

from src.gradient_boosting import my_GradientBoostingRegressor

model = my_GradientBoostingRegressor(
    n_estimators=100,
    lr=0.1,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train,y_train)

train_mse_history = []

for prediction in model.staged_predict(X_train):

    train_mse = mean_squared_error(y_train,prediction)
    train_mse_history.append(train_mse)


val_mse_history = []

for prediction in model.staged_predict(X_val):

    val_mse = mean_squared_error(y_val,prediction)
    val_mse_history.append(val_mse)

# Boosting Round vs MSE

boosting_rounds = np.arange(1,model.n_estimators + 1)

plt.figure(figsize=(8, 5))

plt.plot(boosting_rounds,train_mse_history,label="Train MSE")
plt.plot(boosting_rounds,val_mse_history,label="Validation MSE")

plt.xlabel("Boosting Round")
plt.ylabel("MSE")
plt.title("Gradient Boosting:Boosting Round vs MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Boosting Round vs Residual 

plt.figure(figsize=(8, 5))
plt.plot(boosting_rounds,model.mean_abs_residual_history)
plt.xlabel("Boosting Round")
plt.ylabel("Mean Absolute Residual")
plt.title("Residual Reduction During Boosting")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# sanity check
sklearn_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    loss="squared_error",
    random_state=42
)

sklearn_model.fit(X_train,y_train)

scratch_pred = model.predict(X_val)
sklearn_pred = sklearn_model.predict(X_val)

scratch_mse = mean_squared_error(y_val,scratch_pred)
sklearn_mse = mean_squared_error(y_val,sklearn_pred)


print(
    f"Scratch Validation MSE: "
    f"{scratch_mse:.4f}"
)

print(
    f"sklearn Validation MSE: "
    f"{sklearn_mse:.4f}"
)