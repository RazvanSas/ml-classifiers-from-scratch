# k-Nearest Neighbours from Scratch

Euclidean-distance classification using a heap to retrieve neighbours and majority voting.

## Run

From the repository root, after installing `requirements.txt`:

```bash
python KNearest_Classifier.py
```

## Data and preprocessing

Requires `Social_Network_Ads.csv` in the repository root, with numeric feature columns and the class label in the last column. The scatter plot assumes at least two features and labels 0 and 1. Uses a 75/25 split with random_state=0 and training-fitted standardization.

## Output and configuration

Displays the input scatter plot, then prints a confusion matrix. Close the plot window to continue. The demo uses k=5.

## Implementation scope

Prediction compares each query to every training example. The implementation does not validate k against the training-set size.

The estimator is implemented directly in Python/NumPy; scikit-learn utilities are used where present for datasets, preprocessing, splitting or evaluation, not as the estimator.

[Back to project README](../README.md)

