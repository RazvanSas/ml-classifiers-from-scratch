# Decision Tree Classifier from Scratch

Builds a binary tree recursively, selecting feature thresholds that minimize weighted child entropy and predicting the majority class at leaves.

## Run

From the repository root, after installing `requirements.txt`:

```bash
python decision_tree.py
```

## Data and preprocessing

Uses the breast cancer dataset bundled with scikit-learn and an 80/20 train/test split. No separate CSV is required.

## Output and configuration

Prints accuracy and a confusion matrix. Defaults: maximum depth 5 and minimum split size 2. The split has no fixed random seed.

## Implementation scope

Searches observed feature values as candidate thresholds. Does not implement pruning or missing-value handling.

The estimator is implemented directly in Python/NumPy; scikit-learn utilities are used where present for datasets, preprocessing, splitting or evaluation, not as the estimator.

[Back to project README](../README.md)

