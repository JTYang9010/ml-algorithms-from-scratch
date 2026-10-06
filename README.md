# Machine Learning Algorithms from Scratch

## 1. Motivation
This project was developed to understand the underlying mechanisms of
tree-based machine learning algorithms rather than treating them as
black-box models.

I implemented Decision Tree, Random Forest, Gradient Boosting, and a
simplified XGBoost-style regressor from scratch using Python and NumPy.
The project focuses not only on implementation, but also on analyzing
how model structure, learning strategy, and hyperparameters affect
training behavior and generalization.

## 2. Algorithms Implemented

Decision Tree, 
Random Forest, 
Gradient Boosting, 
XGBoost

## 3. Algorithms and Experimental Analysis

### 3.1 Decision Tree

Implemented a CART-style regression tree from scratch and investigated
how tree complexity affects prediction performance.

**Question:**  
How does tree depth affect model performance and generalization?

**Experiment:**  
A synthetic nonlinear regression dataset was split into training,
validation, and test sets. We varied the maximum tree depth from 1 to 12
and evaluated Train / Validation / Test MSE.

**Result:**
<img width="1920" height="975" alt="CART Regression Tree Depth vs MSE" src="https://github.com/user-attachments/assets/9ee2ed44-6438-4189-8905-97e4091f24ba" />



**Observation:**  
Increasing tree depth consistently reduces training error, while
validation and test error improve rapidly at shallow depths and then
largely plateau. The results illustrate how increasing model complexity
affects training performance and generalization.

### 3.2 Random Forest

Extended the scratch Decision Tree implementation into a Random Forest
by combining bootstrap sampling, random feature selection, and ensemble
prediction.

**Question:**  
How do sampling, feature randomness, and the number of trees affect
ensemble learning?

#### Experiment 1 — Bootstrap Sampling

**Focus:**  
Investigate how bootstrap sampling creates different training subsets
and Out-of-Bag (OOB) samples for individual trees.

**Result:**

<img width="800" height="500" alt="Bootstrap Composition for Each Tree" src="https://github.com/user-attachments/assets/df520814-ff41-431c-9301-3090ea79e5b4" />

**Observation:**  
Each tree is trained on a different bootstrap sample, leaving a subset
of the original training data as OOB samples. This introduces diversity
among individual trees while allowing unused samples to serve as an
independent evaluation set.


#### Experiment 2 — Random Feature Selection

**Focus:**  
Compare trees trained with all features against random feature selection
to examine how feature randomness changes tree structures and predictions.

**Observation:**  
Using random feature selection produces different tree structures across
random seeds, leading to different predictions and validation errors.
This demonstrates how feature randomness introduces diversity among the
trees in a Random Forest.


#### Experiment 3 — Number of Trees

**Experiment:**  
`n_estimators = 1, 10, 50, 100`

**Result:**

<img width="819" height="499" alt="image" src="https://github.com/user-attachments/assets/a728f4ee-53bb-43d1-8400-2fa0817d8787" />

**Observation:**  
Increasing the number of trees substantially reduces prediction error
from 1 to 10 trees, while the improvement becomes smaller as more trees
are added. This indicates diminishing returns as the ensemble grows.

#### Scratch vs. scikit-learn

The scratch implementation was further compared with
scikit-learn's Random Forest under the same model configuration.

**Result:**

<img width="907" height="560" alt="image" src="https://github.com/user-attachments/assets/3b294f67-2feb-4de6-9a06-ff7686f5ec81" />

**Observation:**  
The scratch implementation shows a similar performance trend to
scikit-learn as the number of trees increases, providing a practical
consistency check for the implementation.


### 3.3 Gradient Boosting Study

#### Phase 1 — Boosting Dynamics

Question:
How does sequential boosting reduce prediction error?

**Experiment:**  
A synthetic nonlinear regression dataset was used to track training and
validation MSE over 100 boosting stages.

Result:
<img width="800" height="500" alt="Boosting Round vs MSE" src="https://github.com/user-attachments/assets/caf5a040-c755-4303-a172-7d8291c7e1f0" />

**Observation:**  
Both training and validation error decrease as boosting stages are added,
showing how sequential weak learners progressively improve the model.
The validation error also continues to decrease without a clear increase
within the tested range.

---

#### Phase 2 — Learning Rate

Question:
How does learning rate affect convergence and generalization?

Experiment:
lr = 1, 0.3, 0.1, 0.03

Result:
<img width="900" height="600" alt="Val MSE vs Boosting round under differrent lr" src="https://github.com/user-attachments/assets/043afe67-d423-434a-86b2-a64ea11fdb49" />

**Observation:**  
A larger learning rate leads to faster convergence but can cause the
validation error to increase after the initial improvement. Smaller
learning rates require more boosting stages but provide more gradual
optimization. This illustrates the trade-off between convergence speed
and generalization.

---

#### Phase 3 — Model Complexity

Question:
How does tree depth affect generalization?

Experiment:
depth = 1, 2, 3, 5, 8

Result:
<img width="900" height="600" alt="Gap for differrent tree depth" src="https://github.com/user-attachments/assets/a7f2e2c4-b26f-4a00-9ca7-05e7813d3fa3" />

**Observation:**  
The train-validation gap varies with weak-learner depth and generally
becomes larger as boosting progresses. This shows that the complexity
of the individual weak learners affects the balance between fitting the
training data and generalization.

---

#### Phase 4 — Real Dataset Benchmark

**Question:**  
How does the scratch implementation compare with optimized library
implementations on a real-world regression dataset?

Scratch
vs.
scikit-learn

Metrics:
MSE / RMSE / R² / Training Time

Result:
<img width="757" height="122" alt="image" src="https://github.com/user-attachments/assets/9eec2125-62a8-48c7-adf7-cfb80e00af72" />

**Observation:**  
The scratch implementations achieve comparable predictive performance
to the corresponding scikit-learn models, while requiring substantially
more training time. This highlights the difference between implementing
the core algorithm and optimizing it for practical computation.

### 3.4 XGBoost from Scratch

Implemented a simplified XGBoost regressor from scratch to understand
how gradient-based tree boosting extends the basic Gradient Boosting
framework with second-order information and regularization.

#### Phase 1 — XGBoost Tree

**Question:**  
How does a single XGBoost tree use gradient, Hessian, and gain to determine
its structure?

**Experiment:**  
A synthetic nonlinear regression dataset was used to train a single
XGBoost tree. The implementation computes gradients and Hessians, evaluates
split gain, and determines regularized leaf weights.

**Result:**

<img width="800" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/2402518a-2949-434c-b1e4-f87141501add" />


**Observation:**  
The single tree captures the major structure of the nonlinear target, but
the predictions remain piecewise constant because the model consists of
only one tree.

---

#### Phase 2 — XGBoost Regressor

**Question:**  
How does sequentially combining XGBoost trees improve prediction accuracy?

**Experiment:**  
Built a multi-tree XGBoost regressor using 100 boosting stages with
learning rate = 0.1, max depth = 3, and L2 regularization
lambda = 1.0.

**Result:**

<img width="800" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/d69dd208-5812-424d-a475-0b59b94ba09b" />


**Observation:**  
Compared with the single-tree model, the multi-tree XGBoost regressor
produces predictions that are much closer to the ideal $y=x$ relationship,
showing how sequential boosting progressively improves the approximation
of the nonlinear target.

