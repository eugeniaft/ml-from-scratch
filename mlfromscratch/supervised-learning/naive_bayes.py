import numpy as np


class MultinomialNB():

    def fit(self, X, y, alpha):

        p_0 = X[y == 0].sum(axis=0)
        p_1 = X[y == 1].sum(axis=0)

        l_0 = (p_0 + alpha) / (sum(y == 0) + alpha)
        l_1 = (p_1 + alpha) / (sum(y == 1) + alpha)

        self.r = np.log(l_0/l_1)
        self.b = np.log(sum(y == 1)/sum(y == 0))

    def predict(self, w):

        y_pred = (w @ self.r + self.b) > 0

        return y_pred * 1

    def score(self, y, y_pred):

        acc = (y == y_pred).mean()

        print('Model accuracy: {}'.format(round(acc, 2)))
