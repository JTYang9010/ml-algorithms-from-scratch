import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error,r2_score)

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from src.gradient_boosting import my_GradientBoostingRegressor
from src.random_forest import RandomForestRegressorScratch

# path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# setting

RANDOM_STATE = 42
RUN_SKLEARN_BENCHMARK = True

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "rand_state": 42
}

GB_PARAMS = {
    "n_estimators": 300,
    "lr": 0.3,
    "max_depth": 2,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
}

def load_diabetes_dataset():

    dataset = load_diabetes()

    X = dataset.data
    y = dataset.target

    return X, y

def split_dataset(
    X,
    y
):

    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=RANDOM_STATE
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.15 / 0.85,
            random_state=RANDOM_STATE
        )
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

def calculate_metrics(
    y_true,
    y_pred
):

    mse = mean_squared_error(y_true,y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true,y_pred)

    return (mse,rmse,r2)

def evaluate_scratch_rf(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    model = RandomForestRegressorScratch(**RF_PARAMS)

    start_time = time.perf_counter()

    model.fit(X_train,y_train)

    training_time = (time.perf_counter()- start_time)

    train_pred = model.predict(X_train)

    val_pred = model.predict(X_val)

    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train,train_pred)

    val_mse = mean_squared_error(y_val,val_pred)

    test_mse, test_rmse, test_r2 = (calculate_metrics(y_test,test_pred))

    return {
        "model": "RF Scratch",
        "train_mse": train_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "training_time": training_time,
        "best_stage": None,
        "test_pred": test_pred
    }

def evaluate_scratch_gb(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    model = my_GradientBoostingRegressor(**GB_PARAMS)

    start_time = time.perf_counter()

    model.fit(X_train,y_train)

    training_time = (time.perf_counter()- start_time)

    val_predictions = list(model.staged_predict(X_val))

    val_mse_history = np.array([mean_squared_error(y_val, pred)for pred in val_predictions])

    best_index = np.argmin(val_mse_history)
    best_stage = (best_index+ 1)

    train_predictions = list(model.staged_predict(X_train))
    test_predictions = list(model.staged_predict(X_test))

    train_pred = (train_predictions[best_index])
    val_pred = (val_predictions[best_index])
    test_pred = (test_predictions[best_index])

    train_mse = mean_squared_error(y_train,train_pred)
    val_mse = mean_squared_error( y_val,val_pred)
    test_mse, test_rmse, test_r2 = (calculate_metrics(y_test,test_pred))

    return {
        "model": "GB Scratch",
        "train_mse": train_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "training_time": training_time,
        "best_stage": best_stage,
        "test_pred": test_pred
    }

def evaluate_sklearn_rf(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    model = RandomForestRegressor(
        n_estimators=RF_PARAMS[
            "n_estimators"
        ],
        max_depth=RF_PARAMS[
            "max_depth"
        ],
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    start_time = time.perf_counter()

    model.fit(X_train,y_train)

    training_time = (time.perf_counter()- start_time)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train,train_pred)
    val_mse = mean_squared_error(y_val,val_pred)
    test_mse, test_rmse, test_r2 = (calculate_metrics(y_test,test_pred))

    return {
        "model": "RF sklearn",
        "train_mse": train_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "training_time": training_time,
        "best_stage": None,
        "test_pred": test_pred
    }

def evaluate_sklearn_gb(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    model = GradientBoostingRegressor(
        n_estimators=GB_PARAMS[
            "n_estimators"
        ],
        learning_rate=GB_PARAMS[
            "lr"
        ],
        max_depth=GB_PARAMS[
            "max_depth"
        ],
        min_samples_split=2,
        min_samples_leaf=1,
        loss="squared_error",
        random_state=42
    )

    start_time = time.perf_counter()

    model.fit(X_train,y_train)

    training_time = (time.perf_counter()- start_time)

    val_predictions = list(model.staged_predict(X_val))

    val_mse_history = np.array([mean_squared_error(y_val,pred)for pred in val_predictions])

    best_index = np.argmin(val_mse_history)

    best_stage = (best_index+ 1)

    train_predictions = list(model.staged_predict( X_train))
    test_predictions = list(model.staged_predict(X_test))


    train_pred = (train_predictions[best_index])
    val_pred = (val_predictions[best_index])
    test_pred = (test_predictions[best_index])


    train_mse = mean_squared_error(y_train,train_pred)
    val_mse = mean_squared_error( y_val,val_pred)
    test_mse, test_rmse, test_r2 = (calculate_metrics( y_test,test_pred))

    return {
        "model": "GB sklearn",
        "train_mse": train_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "training_time": training_time,
        "best_stage": best_stage,
        "test_pred": test_pred
    }

dataset_name = "Diabetes"

X, y = load_diabetes_dataset()


print()
print("=" * 80)
print(f"Dataset: {dataset_name}")
print("=" * 80)

print(
    "Dataset shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
) = split_dataset(
    X,
    y
)


print()
print("Data split")
print("-" * 40)

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)

print()
print("=" * 80)
print("Training Scratch Random Forest")
print("=" * 80)

# my rf
rf_result = evaluate_scratch_rf(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
)


rf_result[
    "dataset"
] = dataset_name

print()
print("=" * 80)
print("Training Scratch Gradient Boosting")
print("=" * 80)

# my gb
gb_result = evaluate_scratch_gb(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
)


gb_result[
    "dataset"
] = dataset_name

#sanity check
all_results = [
    rf_result,
    gb_result
]


if RUN_SKLEARN_BENCHMARK:

    print()
    print("=" * 80)
    print("Training sklearn Random Forest")
    print("=" * 80)


    rf_sk_result = evaluate_sklearn_rf(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    rf_sk_result[
        "dataset"
    ] = dataset_name


    print()
    print("=" * 80)
    print("Training sklearn Gradient Boosting")
    print("=" * 80)


    gb_sk_result = evaluate_sklearn_gb(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    gb_sk_result[
        "dataset"
    ] = dataset_name


    all_results.extend(
        [
            rf_sk_result,
            gb_sk_result
        ]
    )

# summary

# =========================================================
# 18. Summary
# =========================================================

print()
print("=" * 125)

print(
    "Phase 4 - Diabetes Regression Summary"
)

print("=" * 125)


print(
    f"{'Model':<18}"
    f"{'Train MSE':<15}"
    f"{'Val MSE':<15}"
    f"{'Test MSE':<15}"
    f"{'Test RMSE':<15}"
    f"{'Test R2':<15}"
    f"{'Time(s)':<15}"
    f"{'Best Stage':<15}"
)


for result in all_results:

    if result[
        "best_stage"
    ] is None:

        best_stage = "-"

    else:

        best_stage = str(
            result[
                "best_stage"
            ]
        )


    print(
        f"{result['model']:<18}"
        f"{result['train_mse']:<15.4f}"
        f"{result['val_mse']:<15.4f}"
        f"{result['test_mse']:<15.4f}"
        f"{result['test_rmse']:<15.4f}"
        f"{result['test_r2']:<15.4f}"
        f"{result['training_time']:<15.3f}"
        f"{best_stage:<15}"
    )