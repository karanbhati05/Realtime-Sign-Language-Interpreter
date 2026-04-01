import numpy as np

class EnsembleModel:
    def __init__(self, mlp, rf, w1=0.6, w2=0.4):
        self.mlp = mlp
        self.rf  = rf
        self.w1  = w1
        self.w2  = w2

    def predict_proba(self, X):
        p1 = self.mlp.predict_proba(X)
        p2 = self.rf.predict_proba(X)
        return self.w1 * p1 + self.w2 * p2

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
