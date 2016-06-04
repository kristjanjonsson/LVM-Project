import numpy as np


class Model:
    '''An abstract model class.
      - self.params must be a dictionary mapping string parameter names to numpy
        arrays containing parameter values.
    '''

    def loss(self, X, y=None, use_reg=True):
        '''
        Inputs:
        - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
        - y: Array of targets, of shape (N,) giving targets for X where y[i] is the
          target for X[i].
        Optional:
        - use_reg: Boolean to turn on/off l2-regularization if the model has it.

        Returns:
        If y is None, run a test-time forward pass and return:
        - scores: Array of shape (N,) giving regression predictions.

        If y is not None, run a training time forward and backward pass and return
        a tuple of:
        - loss: Scalar giving the loss
        - grads: Dictionary with the same keys as self.params mapping parameter
          names to gradients of the loss with respect to those parameters.'''
        raise NotImplementedError()

    def predict(self, X, *args, **kwargs):
        '''Predicts y based on X.'''
        raise NotImplementedError()


class MeanModel(Model):
    '''Model that predicts the mean of the training ratings.'''

    def fit(self, y_train):
        self.mean = y_train.mean()
        self.dtype = y_train.dtype

    def loss(self, X, y=None):
        N = X.shape[0]
        y_predict = self.mean * np.ones(N, dtype=type(self.mean))
        if y is None:
            return y_predict
        else:
            diff = y_predict - y
            loss = np.sum(diff**2) / N
            return loss, None

    def predict(self, X):
        return self.mean * np.ones(len(X), dtype=self.dtype)


class SimpleModel(Model):
    '''Model of the form, r = alpha + beta_u + beta_i,
    where:
    - beta_u: Offset for user u,
    - beta_i: Offset for item i,
    - alpha: Global offset.'''

    def __init__(self, nUsers, nItems, dtype=np.float32):
        self.nUsers = nUsers
        self.nItems = nItems
        self.dtype = dtype
        self.params = {
            'alpha': dtype(0),
            'beta_u': np.zeros(nUsers, dtype=dtype),
            'beta_i': np.zeros(nItems, dtype=dtype)
        }

    def loss(self, X, y=None, **kwargs):
        '''
        Inputs:
        X: Input data of shape (N, 2), [(uId, iId)].
        y: [ratings].
        '''
        alpha = self.params['alpha']
        beta_i = self.params['beta_i']
        beta_u = self.params['beta_u']
        N = len(X)
        users = X[:, 0]
        items = X[:, 1]

        y_predict = alpha + beta_i[items] + beta_u[users]
        if y is None:
            return y_predict

        diff = y_predict - y
        loss = np.sum(diff**2) / N

        dalpha = 2 * diff.sum() / N
        dbeta_i = np.zeros_like(beta_i)
        dbeta_u = np.zeros_like(beta_u)
        for i in range(self.nItems):
            dbeta_i[i] = 2 * diff[items == i].sum() / N
        for u in range(self.nUsers):
            dbeta_u[u] = 2 * diff[users == u].sum() / N
        grads = {
            'alpha': dalpha,
            'beta_u': dbeta_u,
            'beta_i': dbeta_i
        }
        return loss, grads

    def predict(self, X, clipRange=(1.0, 5.0)):
        alpha = self.params['alpha']
        beta_i = self.params['beta_i']
        beta_u = self.params['beta_u']
        users = X[:, 0]
        items = X[:, 1]
        y = alpha + beta_i[items] + beta_u[users]
        low, high = clipRange
        y[y > high] = high
        y[y < low] = low
        assert len(X) == len(y)
        return y


class StandardModel(Model):
    '''Model of the form, r = alpha + beta_u + beta_i + U_u * V_i,
    where:
    - beta_u: Offset for user u,
    - beta_i: Offset for item i,
    - U_u: Latent vector for user u,
    - V_i: Latent vector for item i,
    - alpha: Global offset.'''

    def __init__(self, nUsers, nItems, latentDim=30, reg=0.01, dtype=np.float32):
        self.nUsers = nUsers
        self.nItems = nItems
        self.reg = reg
        self.dtype = dtype
        self.params = {
            'alpha': dtype(0),
            'beta_u': np.zeros(nUsers, dtype=dtype),
            'beta_i': np.zeros(nItems, dtype=dtype),
            'U': np.random.normal(scale=1e-3, size=(nUsers, latentDim)).astype(dtype),
            'V': np.random.normal(scale=1e-3, size=(nItems, latentDim)).astype(dtype)
        }

    def loss(self, X, y=None, use_reg=True):
        '''
        Inputs:
        X: Input data of shape (N, 2), [(uId, iId)].
        y: [ratings].
        '''
        alpha = self.params['alpha']
        beta_u = self.params['beta_u']
        beta_i = self.params['beta_i']
        U = self.params['U']
        V = self.params['V']

        N = len(X)
        users = X[:, 0]
        items = X[:, 1]

        y_predict = alpha + beta_u[users] + beta_i[items] + np.sum(U[users] * V[items], axis=1)
        if y is None:
            return y_predict

        diff = y_predict - y
        loss = np.sum(diff**2) / N

        dalpha = 2 * diff.sum() / N
        dbeta_u = np.zeros_like(beta_u)
        dbeta_i = np.zeros_like(beta_i)
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        for u in range(self.nUsers):
            uIdx = users == u
            uDiff = diff[uIdx]
            dbeta_u[u] = 2 * uDiff.sum() / N
            dU[u] = 2 / N * uDiff.dot(V[items[uIdx]])
        for i in range(self.nItems):
            iIdx = items == i
            iDiff = diff[iIdx]
            dbeta_i[i] = 2 * iDiff.sum() / N
            dV[i] = 2 / N * iDiff.dot(U[users[iIdx]])

        # Regularization.
        if use_reg:
            loss += 0.5 * self.reg * (np.sum(U*U) + np.sum(V*V))
            dU += self.reg * U
            dV += self.reg * V
        grads = {
            'alpha': dalpha,
            'beta_u': dbeta_u,
            'beta_i': dbeta_i,
            'U': dU,
            'V': dV
        }
        return loss, grads


    def predict(self, X, clipRange=(1.0, 5.0)):
        alpha = self.params['alpha']
        beta_u = self.params['beta_u']
        beta_i = self.params['beta_i']
        U = self.params['U']
        V = self.params['V']

        N = len(X)
        users = X[:, 0]
        items = X[:, 1]

        y = alpha + beta_u[users] + beta_i[items] + np.sum(U[users] * V[items], axis=1)
        low, high = clipRange
        y[y > high] = high
        y[y < low] = low
        return y


class PMFModel(Model):
    def __init__(self, nUsers, nItems, latentDim=30, lamU=.1, lamV=.1, dtype=np.float32):
        self.lamU = lamU
        self.lamV = lamV
        self.nUsers = nUsers
        self.nItems = nItems
        self.params = {
            'U': np.random.normal(loc=0, scale=.001, size=(nUsers, latentDim)).astype(dtype),
            'V': np.random.normal(loc=0, scale=.001, size=(nItems, latentDim)).astype(dtype)
        }

    def loss(self, X, y=None, use_reg=True):
        '''
        Inputs:
        X: Input data of shape (N, 2), [(uId, iId)].
        y: [ratings].
        '''

        U = self.params['U']
        V = self.params['V']

        N = len(X)
        users = X[:, 0]
        items = X[:, 1]

        y_predict = np.sum(U[users] * V[items], axis=1)
        if y is None:
            return y_predict

        diff = y_predict - y
        loss = np.sum(diff**2) / N

        # compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        for u in range(self.nUsers):
            uIdx = users == u
            uDiff = diff[uIdx]
            dU[u] = 2.0 / N * uDiff.dot(V[items[uIdx]])
        for i in range(self.nItems):
            iIdx = items == i
            iDiff = diff[iIdx]
            dV[i] = 2.0 / N * iDiff.dot(U[users[iIdx]])

        # Regularization
        if use_reg:
            loss += 0.5 * (self.lamU * np.sum(U*U) + self.lamV * np.sum(V*V))
            dU += self.lamU * U
            dV += self.lamV * V
        grads = {
            'U': dU,
            'V': dV
        }
        return loss, grads


class MixtureModel(Model):

    def __init__(self, nUsers, nItems, nProfiles, latentDim=30, reg=0.01, alpha=None, dtype=np.float32):
        self.nUsers = nUsers
        self.nItems = nItems
        self.nProfiles = nProfiles
        self.reg = reg
        # We fold in the -1 to each alpha term.
        self.alpha = alpha.astype(dtype) - 1 if alpha is not None else np.zeros(nProfiles, dtype=dtype)
        self.dtype = dtype
        self.params = {
            'pi': np.ones(nProfiles, dtype=dtype) / nUsers,
            'U': np.random.normal(scale=1e-3, size=(nProfiles, latentDim)).astype(dtype),
            'V': np.random.normal(scale=1e-3, size=(nItems, latentDim)).astype(dtype)
        }
        Z = np.zeros((nUsers, nProfiles), dtype=dtype)
        Z[np.arange(nUsers), np.random.choice(np.arange(nProfiles), size=nUsers)] = 1.0
        self.hiddenState = {
            'Z': Z
        }
        # self.hiddenState['Z'] /= self.hiddenState['Z'].sum(axis=1).reshape(nUsers, 1)

    def updatePi(self):
        Z = self.hiddenState['Z']
        pi = Z.sum(axis=0) + self.alpha
        pi /= pi.sum()
        self.params['pi'] = pi

    def estimate(self, X, y):
        '''Updates the hiddenState based on current parameters.'''
        Z = self.hiddenState['Z']
        U = self.params['U']
        V = self.params['V']
        pi = self.params['pi']

        N = len(X)
        users = X[:, 0]
        items = X[:, 1]

        scores = V[items].dot(U.T)
        diff = scores - y.reshape((N, 1))
        losses = diff**2
        logp = np.log(pi) - 0.5 * losses.sum(axis=0)
        self.params['Z'] = softmax(logp)


    def loss(self, X, y=None, use_reg=True):
        '''
        Inputs:
        X: Input data of shape (N, 2), [(uId, iId)].
        y: [ratings].

        Returns:
        Loss with respect U and V given the current estimate of hiddenState.
        '''
        Z = self.hiddenState['Z']
        U = self.params['U']
        V = self.params['V']

        N = len(X)
        users = X[:, 0]
        items = X[:, 1]


        if y is None:
            raise NotImplementedError("EM doesn't really fit this API.")

        # scores.shape = (N, P)
        # scores[i] is the predicted rating for all P profiles.
        scores = V[items].dot(U.T)
        diff = scores - y.reshape((N, 1))

        # We have to average across the hiddenState probabilities.
        losses = diff**2
        losses *= Z[users]
        loss = np.sum(losses) / N

        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        weightedDiff = 2 * diff * Z[users] / N
        dU += weightedDiff.T.dot(V[items])

        for i in range(self.nItems):
            iIdx = items == i
            iDiff = weightedDiff[iIdx]
            dV[i] = iDiff.dot(U).sum(axis=0)

        # Regularization.
        if use_reg:
            loss += 0.5 * self.reg * (np.sum(U*U) + np.sum(V*V))
            dU += self.reg * U
            dV += self.reg * V
        grads = {
            'U': dU,
            'V': dV
        }
        return loss, grads

    def predict(self, X, mean=0.0, clipRange=(1.0, 5.0)):
        '''Adds mean to the model prediction and clips the values to be in clipRange.'''
        Z = self.hiddenState['Z']
        U = self.params['U']
        V = self.params['V']

        users = X[:, 0]
        items = X[:, 1]
        # user2profile = np.argmax(Z, axis=1)
        # profiles = user2profile[users]

        # y = np.sum(V[items] * U[profiles], axis=1)
        scores = V[items].dot(U.T)
        scores *= Z[users]
        y = scores.sum(axis=1)

        y += mean
        low, high = clipRange
        y[y > high] = high
        y[y < low] = low
        return y




def softmax(x):
    '''x has shape (n, )'''
    r = x.copy()
    r -= r.max()
    r = np.exp(r)
    r /= r.sum()
    return r

