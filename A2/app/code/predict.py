"""Prediction for the A2 model."""

import numpy as np


def predict_price(raw_df, art):
    """Predicts selling price in rupees from a dataframe of raw car details."""

    X = art['preprocessor'].transform(raw_df)

    if art['polynomial']:
        k = art['n_numeric']
        num, cat = X[:, :k], X[:, k:]

        # Pairwise products of the numeric columns, rescaled with the training statistics.
        products = np.hstack([(num[:, i] * num[:, j]).reshape(-1, 1)
                              for i in range(k) for j in range(i, k)])
        products = (products - art['poly_mean']) / art['poly_std']
        X = np.hstack([num, products, cat])

    # The bias travels as a constant input of one.
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # The model was trained on the centred target, so the mean is added back.
    log_price = X @ art['theta'] + art['y_mean']
    return np.exp(log_price)