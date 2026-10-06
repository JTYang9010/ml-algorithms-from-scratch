import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(project_root)


from src.xgboost_regressor import MyXGBRegressor



# ==========================
# Dataset
# ==========================


rng=np.random.default_rng(42)


X=rng.uniform(
    -1,
    1,
    (1000,8)
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


y=(
    true_function(X)
    +
    rng.normal(
        0,
        0.1,
        len(X)
    )
)



X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ==========================
# Train XGBoost
# ==========================


model=MyXGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    lambda_reg=1
)



model.fit(
    X_train,
    y_train
)



# ==========================
# Evaluation
# ==========================


y_pred=model.predict(
    X_test
)


mse=mean_squared_error(
    y_test,
    y_pred
)


print(
    "Test MSE:",
    mse
)



# ==========================
# Plot
# ==========================


plt.figure(
    figsize=(8,5)
)


plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)


min_val=min(
    np.min(y_test),
    np.min(y_pred)
)

max_val=max(
    np.max(y_test),
    np.max(y_pred)
)


plt.plot(
    [min_val,max_val],
    [min_val,max_val],
    linestyle="--",
    label="y=x"
)


plt.xlabel(
    "True y"
)

plt.ylabel(
    "Prediction"
)

plt.title(
    "XGBoost Phase 2 Prediction"
)

plt.legend()

plt.grid(alpha=0.3)

plt.show()