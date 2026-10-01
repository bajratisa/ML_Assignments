import numpy as np
import pytest

from app.logistic_regression import LogisticRegression

# The web app feeds the model the output of the preprocessing pipeline, which has 41 columns.
N_FEATURES = 41
N_CLASSES = 4


def make_fitted_model(n_rows=200):
    """A small model on made-up data. No MLflow and no Cars.csv, so the tests run anywhere."""
    rng = np.random.default_rng(42)   # fixed seed, so the data (and any failure) is repeatable
    X = rng.normal(size=(n_rows, N_FEATURES))
    y = rng.integers(0, N_CLASSES, size=n_rows)
    assert set(np.unique(y)) == set(range(N_CLASSES))

    # A small learning rate and a few hundred steps are plenty: these tests check the shapes
    # and the input handling, not how accurate the model is.
    model = LogisticRegression(k=N_CLASSES, lr=0.1, num_iter=300, print_every=0)
    model.fit(X, y)
    return model, X


def test_model_accepts_expected_input():
    # Checks that a fitted model takes the array shape the app will send (n, 41) and that
    # a different number of columns is refused. It matters because a silent mismatch would
    # give wrong prices, and the app can only show a clear error if the model raises one.
    model, X = make_fitted_model()

    # The right shape must run without an error, for the whole array and for a single row.
    model.predict(X)
    model.predict_proba(X)
    model.predict(X[:1])
    model.predict_proba(X[:1])

    # Today the class does not check the number of columns itself. The error comes from
    # numpy, which cannot multiply the input by the weights when the sizes differ.
    # It is a ValueError, so the test only relies on that.
    for wrong_n_features in (N_FEATURES - 1, N_FEATURES + 1):
        # Cut or stretch the real data to one column less or one column more.
        X_wrong = np.hstack([X, X])[:, :wrong_n_features]
        with pytest.raises(ValueError):
            model.predict(X_wrong)
        with pytest.raises(ValueError):
            model.predict_proba(X_wrong)


def test_output_has_expected_shape():
    # Checks the shape and content of the outputs: one integer class 0 to 3 per row, and
    # one probability per class per row that adds up to 1. It matters because the app turns
    # the class number into a price range and shows the probabilities, so a missing column
    # or probabilities that do not add up to 1 would break or mislead it.
    model, X = make_fitted_model()
    n = X.shape[0]

    predictions = model.predict(X)
    assert predictions.shape == (n,)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert predictions.min() >= 0 and predictions.max() <= N_CLASSES - 1

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (n, N_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
