import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris

def add_bias_feature(a):
    a_extended = np.zeros((a.shape[0], a.shape[1] + 1))
    a_extended[:, :-1] = a
    a_extended[:, -1] = int(1)
    return a_extended

class CustomSVM(object):

    def __init__(self, etha=0.01, alpha=0.1, epochs=200):
        self._epochs = epochs
        self._etha = etha
        self._alpha = alpha
        self._w = None
        self.history_w = []
        self.train_errors = None
        self.val_errors = None
        self.train_loss = None
        self.val_loss = None

    def fit(self, X_train, Y_train, X_val, Y_val, verbose=False):

        if len(set(Y_train)) != 2 or len(set(Y_val)) != 2:
            raise ValueError("Number of classes in Y is not equal 2!")
        
        X_train = add_bias_feature(X_train)
        X_val = add_bias_feature(X_val)
        self._w = np.random.normal(loc=0, scale=0.05, size=X_train.shape[1])
        self.history_w.append(self._w)
        train_errors = []
        val_errors = []
        train_loss_epoch = []
        val_loss_epoch = []

        for epoch in range(self._epochs):
            tr_err = 0
            val_err = 0
            tr_loss = 0
            val_loss = 0
            for i, x in enumerate(X_train):
                margin = Y_train[i] * np.dot(self._w, X_train[i])
                if margin >= 1:
                    self._w = self._w - self._etha * self._alpha * self._w / self._epochs
                    tr_loss += self.soft_margin_loss(X_train[i], Y_train[i])
                else:
                    self._w = self._w + self._etha * (Y_train[i] * X_train[i] - self._alpha * self._w / self._epochs)
                    tr_err += 1
                    tr_loss += self.soft_margin_loss(X_train[i], Y_train[i])
                self.history_w.append(self._w)
            for i, x in enumerate(X_val):
                val_loss += self.soft_margin_loss(X_val[i], Y_val[i])
                val_err += (Y_val[i] * np.dot(self._w, X_val[i]) < 1).astype(int)
            if verbose:
                print("epoch {}. Errors={}. Mean Hinge_loss ={}".format(epoch, err, loss))
            train_errors.append(tr_err)
            val_errors.append(val_err)
            train_loss_epoch.append(tr_loss)
            val_loss_epoch.append(val_loss)
        self.history_w = np.array(self.history_w)
        self.train_errors = np.array(train_errors)
        self.val_errors = np.array(val_errors)
        self.train_loss = np.array(train_loss_epoch)
        self.val_loss = np.array(val_loss_epoch)

    def predict(self, X:np.array) -> np.array:
        y_pred = []
        X_extended = add_bias_feature(X)
        for i in range(len(X_extended)):
            y_pred.append(np.sign(np.dot(self._w, X_extended[i])))
        return np.array(y_pred)
    
    def hinge_loss(self, x, y):
        return max(0, 1 - y * np.dot(x, self._w))
    
    def soft_margin_loss(self, x, y):
        return self.hinge_loss(x, y) + self._alpha * np.dot(self._w, self._w)

iris = load_iris()
X = iris.data
Y = iris.target

pca = PCA(n_components=2)
X = pca.fit_transform(X)
Y = (Y > 0).astype(int) * 2 - 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=2020)

svm = CustomSVM(etha=0.005, alpha=0.006, epochs=150)
svm.fit(X_train, Y_train, X_test, Y_test)

print(svm.train_errors)