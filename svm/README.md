# Support Vector Machine from Scratch

A linear soft-margin SVM implemented using the Lagrangian method and its dual formulation. Sequential Minimal Optimization (SMO) updates pairs of Lagrange multipliers while respecting box and equality constraints. KKT checks identify candidates for updates.

## Lagrangian dual formulation

For labels $y_i \in \{-1,+1\}$, optimize:

$$
\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j)
$$

subject to $0 \leq \alpha_i \leq C$ and $\sum_i \alpha_i y_i=0$.

The linear kernel is $K(x_i,x_j)=x_i^T x_j$. Predictions use the sign of $\sum_i \alpha_i y_i K(x_i,x)+b$. SMO performs pairwise constrained updates; the implementation also reconstructs $w=\sum_i \alpha_i y_i x_i$ for plotting.

## Run

From the repository root, after installing `requirements.txt`:

```bash
python SVM.py
```

## Data and preprocessing

Runs on nine embedded 2D points with labels -1 and +1. No CSV is needed for the default demo. A CSV-loading example is present but commented out.

## Output and configuration

Prints multipliers, bias, weight vector, margin width and training predictions; plots the decision boundary, margins and support vectors. Defaults: C=1, KKT tolerance=0.001, update epsilon=1e-5.

## Implementation scope

The kernel is linear. The default example evaluates training predictions, not held-out accuracy. Pair selection is simplified and there is no hard total-iteration cap.

The estimator is implemented directly in Python/NumPy; scikit-learn utilities are used where present for datasets, preprocessing, splitting or evaluation, not as the estimator.

[Back to project README](../README.md)

