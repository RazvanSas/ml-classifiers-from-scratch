# Gaussian Naive Bayes from Scratch

Estimates per-class means, variances and prior probabilities, then combines Gaussian log-likelihoods with log priors to predict a class.

## Run

From the repository root, after installing `requirements.txt`:

```bash
python gaussian_NB.py
```

## Data and preprocessing

Uses the breast cancer dataset bundled with scikit-learn and an 80/20 train/test split. No separate CSV is required.

## Output and configuration

Prints accuracy and a confusion matrix. The split has no fixed random seed, so scores can vary between runs.

## Implementation scope

Assumes conditional feature independence and Gaussian feature distributions. Variance smoothing is not implemented, so zero-variance features require care.

The estimator is implemented directly in Python/NumPy; scikit-learn utilities are used where present for datasets, preprocessing, splitting or evaluation, not as the estimator.

[Back to project README](../README.md)

