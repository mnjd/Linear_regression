import numpy as np

class Linear_regression():

    def __init__(self, mode="norm_eq", lr=0.01, beta=0.9, batch_size=32, epoch=10, max_iter=50000, tol=1e-4):
        if mode not in ["norm_eq", "GD", "SGD"]:
            raise ValueError(mode + " is not a valid choice.")
        self.mode = mode
        self.lr = lr
        self.beta = beta
        self.batch_size = batch_size
        self.epoch = epoch
        self.max_iter = max_iter
        self.tol = tol
        self.W = None
        self.Z = None

    def fit(self, X, Y):
        X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)
        
        ############################
        # Normal equation solution #
        ############################
        
        if self.mode == "norm_eq":
            self.W = np.linalg.inv(X.T.dot(X)).dit(X.T.dot(Y))
            return True

        ####################
        # Gradient Descent #
        ####################

        elif self.mode == "GD":
            it = 0
            if self.W is None:
                self.W = np.zeros((X.shape[1], 1))
            grad = (X.T.dot(X.dot(self.W) - Y)) / X.shape[0]
            while it < self.max_iter and np.linalg.norm(grad) > self.tol:
                it += 1
                self.W -= self.lr * grad
                grad = (X.T.dot(X.dot(self.W) - Y)) / X.shape[0]
            return True

        ##################################
        # Gradient Descent With Momentum #
        ##################################

        elif self.mode == "GDWM":
            it = 0
            if self.W is None:
                self.W = np.zeros((X.shape[1], 1))
                self.Z = np.zeros_like(self.W)
            grad = (X.T.dot(X.dot(self.W) - Y)) / X.shape[0]
            while it < self.max_iter and np.linalg.norm(grad) > self.tol:
                it += 1
                self.Z = self.beta * self.Z + grad
                self.W -= self.lr * self.Z
                grad = (X.T.dot(X.dot(self.W) - Y)) / X.shape[0]
            return True
        
        ###############################
        # Stochastic Gradient Descent #
        ###############################

        elif self.mode == "SGD":
            if self.W is None:
                self.W = np.zeros((X.shape[1], 1))

            n = X.shape[0]
            n_batch = int(n / self.batch_size)
            if n % n_batch != 0:
                n_batch += 1
            j = 0

            for e in range(self.epoch):
                perm = np.random.permutation(n_batch)
                while j < n_batch:
                    i = perm[j]
                    x = X[i*self.batch_size:(i+1)*self.batch_size,:]
                    y = Y[i*self.batch_size:(i+1)*self.batch_size,:]
                    grad = (x.T.dot(x.dot(self.W) - y)) / x.shape[0]
                    self.W -= self.lr * grad
                    j += 1
                j = 0
            return True

        #############################################
        # Stochastic Gradient Descent With Momentum #
        #############################################

        elif self.mode == "SGDWM":
            if self.W is None:
                self.W = np.zeros((X.shape[1], 1))
                self.Z = np.zeros_like(self.W)

            n = X.shape[0]
            n_batch = int(n / self.batch_size)
            if n % n_batch != 0:
                n_batch += 1
            j = 0

            for e in range(self.epoch):
                perm = np.random.permutation(n_batch)
                while j < n_batch:
                    i = perm[j]
                    x = X[i*self.batch_size:(i+1)*self.batch_size,:]
                    y = Y[i*self.batch_size:(i+1)*self.batch_size,:]
                    grad = (x.T.dot(x.dot(self.W) - y)) / x.shape[0]
                    self.Z = self.beta * self.Z + grad
                    self.W -= self.lr * self.Z
                    j += 1
                j = 0
            return True


    def predict(self, X):
        if self.W is None:
            return False
        else:
            X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)
            return X.dot(self.W)
