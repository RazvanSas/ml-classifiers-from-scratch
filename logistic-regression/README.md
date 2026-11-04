# Logistic Regression from Scratch

Binary logistic regression with a sigmoid probability model, vectorized batch gradient descent, an intercept term and gradient-norm stopping.

## Run

From the repository root, after installing `requirements.txt`:

```bash
python logistic_regression.py
```

## Data and preprocessing

Requires `Social_Network_Ads.csv` in the repository root. All columns except the last must be numeric features; the last must contain binary 0/1 labels. The script uses a 75/25 split with random_state=0 and fits StandardScaler only on training data.

## Output and configuration

Prints accuracy and a confusion matrix. Defaults: 100 gradient steps, learning rate 0.1, tolerance 1e-7 and probability threshold 0.5.

## Implementation scope

The sigmoid is not clipped for extreme inputs, and no regularization is implemented.

The estimator is implemented directly in Python/NumPy; scikit-learn utilities are used where present for datasets, preprocessing, splitting or evaluation, not as the estimator.

[Back to project README](../README.md)

