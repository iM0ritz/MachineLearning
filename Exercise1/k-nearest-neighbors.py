"""
Implement the K-Nearest Neighbour algorithm for classification and regression
from scratch and verify that you achieve comparable results to the scikit-learn
implementation of the algorithm, using different K values, for various standard
datasets available through scikit-learn. Use cross-validation to compute
average performance scores. Use accuracy (for classification) and mean squared
error (for regression) to compute performance scores.
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error

class KNearestNeighbors:
    def __init__(self, n_neighbors=3, is_regressor=None):
        """
        is_regressor: if None, infer from y in fit(); otherwise force behavior.
        """
        self.n_neighbors = n_neighbors
        self._is_regressor = is_regressor

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        # decide if we should treat this as a regressor (allow forcing via ctor)
        if self._is_regressor is None:
            # treat as regression if target is multi-output or float-valued
            if getattr(y, "ndim", 1) > 1 or np.issubdtype(y.dtype, np.floating):
                self.is_regressor = True
            else:
                # for integer 1D targets, also check unique ratio (similar to sklearn heuristic)
                n_unique = np.unique(y).size
                self.is_regressor = (n_unique > 0.5 * y.shape[0])
        else:
            self.is_regressor = bool(self._is_regressor)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        if getattr(self, "is_regressor", False):
            # predictions may be scalar or vector; stack appropriately
            return np.vstack(predictions) if np.asarray(predictions[0]).ndim > 0 else np.array(predictions)
        else:
            return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # Extract the labels of the k nearest neighbor
        k_nearest_labels = np.asarray([self.y_train[i] for i in k_indices])
        if getattr(self, "is_regressor", False):
            # regression -> return mean (works for multi-output as well)
            return np.mean(k_nearest_labels, axis=0)
        else:
            # classification -> ensure integer labels for bincount
            labels_int = k_nearest_labels.astype(int)
            return np.bincount(labels_int).argmax()

# Example usage:
if __name__ == "__main__":
    # Load datasets
    dataset_list = []
    dataset_list.append(datasets.load_iris())
    dataset_list.append(datasets.load_diabetes())
    dataset_list.append(datasets.load_digits())
    dataset_list.append(datasets.load_linnerud())
    dataset_list.append(datasets.load_wine())

    for dataset in dataset_list:
        X, y = dataset.data, dataset.target
        # short dataset name: pick first non-empty DESCR line that's not a reST reference
        def _dataset_name(ds):
            descr = getattr(ds, "DESCR", "") or ""
            for line in descr.splitlines():
                s = line.strip()
                if s and not s.startswith(".."):
                    return s
            return getattr(ds, "filename", "dataset")
        dataset_name = _dataset_name(dataset)

        # choose folder and metric depending on target type
        # simple heuristic: treat as classification only for 1-D integer labels
        # (avoids misclassifying regression targets that happen to have few unique values)
        n_splits = 5
        is_classification = (y.ndim == 1) and np.issubdtype(y.dtype, np.integer)

        if is_classification:
            class_counts = np.bincount(y.astype(int))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            metric_name = "accuracy"
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            metric_name = "mse"

        k = 5
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            # force the correct mode per dataset so folds aren't mis-inferred
            knn = KNearestNeighbors(n_neighbors=k, is_regressor=(not is_classification))
            knn.fit(X[train_idx], y[train_idx])
            preds = knn.predict(X[test_idx])
            if is_classification:
                scores.append(accuracy_score(y[test_idx], preds))
            else:
                scores.append(mean_squared_error(y[test_idx], preds))

        scores = np.array(scores)
        print(f"Cross-val {metric_name} own (dataset: {dataset_name}) (k={k}): {scores.mean():.4f} ± {scores.std():.4f}")

        # Compare with scikit-learn's KNeighborsClassifier / KNeighborsRegressor
        if is_classification:
            from sklearn.neighbors import KNeighborsClassifier as SKKNN
        else:
            from sklearn.neighbors import KNeighborsRegressor as SKKNN

        sklearn_scores = []
        for train_idx, test_idx in cv.split(X, y):
            clf = SKKNN(n_neighbors=k)
            # sklearn requires 1D y for classification, reshape for regressors automatically
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            if is_classification:
                sklearn_scores.append(accuracy_score(y[test_idx], preds))
            else:
                sklearn_scores.append(mean_squared_error(y[test_idx], preds))

        sklearn_scores = np.array(sklearn_scores)
        print(f"Cross-val {metric_name} scikit-learn (dataset: {dataset_name}) (k={k}): {sklearn_scores.mean():.4f} ± {sklearn_scores.std():.4f}")