import numpy as np


class RidgePenalty:
    """Ridge (L2) penalty: lambda * sum of squared weights, and its derivative."""

    def __init__(self, l):
        self.l = l  # lambda value, the strength of the penalty

    def __call__(self, W):
        # lambda * sum(w^2). Squaring makes every weight cost something, and big weights
        # cost much more than small ones, so the model prefers many small weights.
        return self.l * np.sum(np.square(W))

    def derivation(self, W):
        # The derivative of lambda * w^2 is 2 * lambda * w. It always points the same way as
        # the weight, so subtracting it pulls every weight towards zero.
        return self.l * 2 * W


class LogisticRegression:
    """Multinomial logistic regression (softmax) trained with batch gradient descent."""

    def __init__(self, k=4, lr=0.1, num_iter=1000, print_every=100,
                 use_penalty=False, lambda_=0.0):
        self.k = k                      # number of classes
        self.lr = lr                    # learning rate, the size of each update step
        self.num_iter = num_iter        # how many gradient descent steps to take
        self.print_every = print_every  # print the loss every this many steps (0 = never)

        # Ridge (L2) penalty. It is off by default, and then nothing below changes.
        # lambda_ has a trailing underscore because lambda is a Python keyword.
        #
        # Convention used here:
        #   loss = mean cross entropy over the m samples + lambda_ * sum(weights ** 2)
        #   grad = X.T @ (h - Y) / m               + 2 * lambda_ * weights
        # The cross entropy part is averaged over the samples, but the penalty is NOT divided
        # by m. So lambda_ is measured on the scale of the average loss: lambda_ = 0.1 means
        # "0.1 times the squared weights, next to an average loss per car". The assignment
        # formula adds lambda * sum(theta^2) to a loss that is summed (not averaged) over
        # the samples, so the same strength there would be lambda_ * m.
        # The bias weights (the first row of W) are never penalized: the bias only moves
        # the scores up or down and does not make the model more complex or more wiggly.
        if lambda_ < 0:
            raise ValueError("lambda_ must not be negative")
        self.use_penalty = use_penalty
        self.lambda_ = lambda_
        self.penalty = RidgePenalty(lambda_)

    def _add_bias(self, X):
        # The bias is the intercept. A column of ones lets it be learned as one more weight,
        # so the model is X @ W instead of X @ W + b. Doing it here means the caller never
        # has to remember it, and predict adds it in exactly the same way as fit.
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def _one_hot(self, y):
        # y holds class numbers such as [2, 0, 3]. The loss and the gradient need one row per
        # sample with a 1 in the true class column, e.g. 2 -> [0, 0, 1, 0].
        Y = np.zeros((y.shape[0], self.k))
        Y[np.arange(y.shape[0]), y] = 1
        return Y

    def softmax(self, scores):
        # Softmax turns the k raw scores of each sample into k probabilities that are all
        # positive (because of exp) and add up to 1 (because of the division).
        # Subtracting the row maximum first does not change the result, but it stops exp()
        # from overflowing when a score is large.
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def cross_entropy(self, Y, h):
        # Cross entropy loss: -mean( sum over classes of Y * log(h) ).
        # Y is 1 only for the true class, so each sample contributes -log(probability given
        # to the true class). A confident right answer costs almost 0, and a confident wrong
        # answer costs a lot. The clip keeps log() away from exactly 0.
        h = np.clip(h, 1e-15, 1.0)
        return -np.sum(Y * np.log(h)) / Y.shape[0]

    def fit(self, X, y):
        X = self._add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y).astype(int)
        if y.min() < 0 or y.max() >= self.k:
            raise ValueError(f"y must contain integer labels from 0 to {self.k - 1}")

        m, n = X.shape                  # m samples, n weights per class (features + bias)
        Y = self._one_hot(y)            # shape (m, k)

        # One column of weights per class, shape (n, k). Starting from zeros is fine here:
        # this loss has a single bottom, so every start ends up at the same answer.
        self.W = np.zeros((n, self.k))
        self.losses = []                # the loss that is minimized (includes the penalty)
        self.ce_losses = []             # the cross entropy part only

        for i in range(self.num_iter):
            h = self.softmax(X @ self.W)             # predicted probabilities, shape (m, k)
            ce_loss = self.cross_entropy(Y, h)
            self.ce_losses.append(ce_loss)

            loss = ce_loss
            if self.use_penalty:
                # W[1:] skips the first row, which holds the bias weights.
                loss = ce_loss + self.penalty(self.W[1:])
            self.losses.append(loss)

            # Gradient of the loss with respect to W. Softmax and cross entropy cancel
            # nicely, leaving (h - Y): the gap between predicted and true probabilities.
            # X.T @ (h - Y) adds that gap up for every feature, and dividing by m takes
            # the mean, so the step size does not depend on the number of samples.
            grad = X.T @ (h - Y) / m

            if self.use_penalty:
                # The penalty adds its own derivative, 2 * lambda_ * w, to the gradient of every
                # weight except the bias. So each step also pulls the weights towards zero.
                grad[1:] = grad[1:] + self.penalty.derivation(self.W[1:])

            # Move the weights a small step against the gradient to lower the loss.
            self.W = self.W - self.lr * grad

            if self.print_every and i % self.print_every == 0:
                print(f"Loss at iteration {i}: {loss:.4f}")

        return self

    def predict_proba(self, X):
        # Probability of each of the k classes for every row, shape (m, k).
        X = self._add_bias(np.asarray(X, dtype=float))
        return self.softmax(X @ self.W)

    def predict(self, X):
        # The predicted class is the one with the highest probability.
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------------------------
    # Classification metrics, written with plain numpy.
    # y_true and y_pred are 1D arrays of class numbers 0 to k-1. None of these methods use
    # the trained weights, so they also work on any pair of label arrays.
    # ------------------------------------------------------------------------------------

    def _counts(self, y_true, y_pred, c):
        # For one class c, every car falls in exactly one of these groups. Class c is treated
        # as "positive" and every other class as "negative".
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = np.sum((y_pred == c) & (y_true == c))   # true positive: said c, and it is c
        fp = np.sum((y_pred == c) & (y_true != c))   # false positive: said c, but it is not c
        fn = np.sum((y_pred != c) & (y_true == c))   # false negative: said not c, but it is c
        return tp, fp, fn

    def accuracy(self, y_true, y_pred):
        # correct predictions / all predictions. One number for the whole model, so it does
        # not show which class the mistakes are in. That is why the per class scores exist.
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.sum(y_true == y_pred) / y_true.shape[0])

    def precision(self, y_true, y_pred, c):
        # TP / (TP + FP): of all the cars the model called class c, how many really are c.
        # TP + FP is the number of times the model predicted c. If it never predicted c the
        # division would be 0/0, so we return 0 instead.
        tp, fp, fn = self._counts(y_true, y_pred, c)
        if tp + fp == 0:
            return 0.0
        return float(tp / (tp + fp))

    def recall(self, y_true, y_pred, c):
        # TP / (TP + FN): of all the cars that really are class c, how many the model found.
        # TP + FN is the number of true class c cars. If there are none, return 0.
        tp, fp, fn = self._counts(y_true, y_pred, c)
        if tp + fn == 0:
            return 0.0
        return float(tp / (tp + fn))

    def f1(self, y_true, y_pred, c):
        # 2 * P * R / (P + R): the harmonic mean of precision and recall. It is high only when
        # both are high, so a model cannot hide a very low recall behind a high precision.
        # If both are 0 the division would be 0/0, so we return 0.
        p = self.precision(y_true, y_pred, c)
        r = self.recall(y_true, y_pred, c)
        if p + r == 0:
            return 0.0
        return float(2 * p * r / (p + r))

    # Macro average: the plain mean over the k classes, so every class counts the same no
    # matter how many cars it has.

    def macro_precision(self, y_true, y_pred):
        return float(np.mean([self.precision(y_true, y_pred, c) for c in range(self.k)]))

    def macro_recall(self, y_true, y_pred):
        return float(np.mean([self.recall(y_true, y_pred, c) for c in range(self.k)]))

    def macro_f1(self, y_true, y_pred):
        return float(np.mean([self.f1(y_true, y_pred, c) for c in range(self.k)]))

    # Weighted average: each class score is multiplied by the share of true cars in that
    # class (its support / all cars) before adding. Big classes count more, which matches
    # the overall performance better when the classes are not the same size.

    def _weights(self, y_true):
        y_true = np.asarray(y_true)
        return np.array([np.sum(y_true == c) for c in range(self.k)]) / y_true.shape[0]

    def weighted_precision(self, y_true, y_pred):
        scores = np.array([self.precision(y_true, y_pred, c) for c in range(self.k)])
        return float(np.sum(self._weights(y_true) * scores))

    def weighted_recall(self, y_true, y_pred):
        scores = np.array([self.recall(y_true, y_pred, c) for c in range(self.k)])
        return float(np.sum(self._weights(y_true) * scores))

    def weighted_f1(self, y_true, y_pred):
        scores = np.array([self.f1(y_true, y_pred, c) for c in range(self.k)])
        return float(np.sum(self._weights(y_true) * scores))
