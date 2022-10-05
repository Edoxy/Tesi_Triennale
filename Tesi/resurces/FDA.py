import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score


class KPCA_FDA:
    # Fischer Discriminant Analysis
    def __init__(
        self,
        pca_components=None,
        fda_components=None,
        kernel='linear',
        k_degree=3,
        k_gamma=None,
        k_coef0=1
    ):

        self.fda_components = fda_components
        self.pca_components = pca_components
        self.kernel = kernel
        self.k_degree = k_degree
        self.k_gamma = k_gamma
        self.k_coef0 = k_coef0

        return

    def _mean_class(self, X_class):

        mean = np.mean(np.concatenate(X_class, 0), 0)
        mean_class = np.array([np.mean(xc, 0) for xc in X_class])
        return mean_class, mean

    def _scatter_within(self, X_class, mean_class):
        n_features = mean_class.shape[1]
        S_w = np.zeros((n_features, n_features))

        for xc, mc in zip(X_class, mean_class):
            tmp = xc - mc[np.newaxis, :]
            S_w += np.matmul(tmp.T, tmp)

        return S_w

    def _scatter_total(self, X, mean):
        tmp = X - mean[np.newaxis, :]
        S_t = np.matmul(tmp.T, tmp)
        return S_t

    def _sparability(self, eigen_vc, eigen_vl, S_b, S_w):
        w = np.expand_dims(eigen_vc.T, 1)
        a = np.squeeze(np.matmul(np.matmul(w, S_b),
                       np.transpose(w, [0, 2, 1])), (1, 2))
        b = np.squeeze(np.matmul(np.matmul(w, S_w),
                       np.transpose(w, [0, 2, 1])), (1, 2))

        idx_nonzero = np.array(np.nonzero(b))[0]
        a = a[idx_nonzero]
        b = b[idx_nonzero]
        sep = a / b
        eigen_vl, eigen_vc = eigen_vl[idx_nonzero], eigen_vc[:, idx_nonzero]
        idx = sep.argsort()[::-1]

        return eigen_vl[idx], eigen_vc[:, idx]

    def fit(self, X, y, **params):
        # X Dataset
        # y labels
        # Accepts dictionary to conform to sklearn class so that its compatible with GridSearchCV

        # All unique labels
        unique_y = np.unique(y)

        # PCA Transformation
        if self.pca_components == None:
            self.pca_components = X.shape[0] - unique_y.shape[0]
            # this prevents the singularity of the Scatter_within matrix the need to be inverted

        self.pca = KernelPCA(
            n_components=self.pca_components,
            kernel=self.kernel,
            degree=self.k_degree,
            gamma=self.k_gamma,
            coef0=self.k_coef0)
        # Transform the data in the new subspace
        X = self.pca.fit_transform(X)

        # setting the number of components at the maximum possible if None value is set
        if self.fda_components == None:
            self.fda_components = np.min([X.shape[1], len(unique_y)-1])

        # Rearrage the data in class order
        X_class = [X[y == y_class] for y_class in unique_y]
        # Calculate the total mean and the classes mean
        self.m_class, self.mean = self._mean_class(X_class)
        # Calculate the total Scatter matrix
        self.S_t = self._scatter_total(X, self.mean)
        # Calculate of the scatter within matrix
        self.S_w = self._scatter_within(X_class, self.m_class)

        # Calculate the Scatter Between matrix using the simple subtraction
        self.S_b = self.S_t - self.S_w

        # Calculate the psudo_invers of the scatter within matrix
        self.i_S_w = np.linalg.pinv(self.S_w)

        sigma = np.matmul(self.i_S_w, self.S_b)

        # Eigenvalues and eigenvectors solution
        eigen_vl, eigen_vc = np.linalg.eig(sigma)

        # cosa fa?
        eigen_vl, eigen_vc = self._sparability(eigen_vc, eigen_vl, self.S_b, self.S_w)

        # Transformation Matrix that projects in the new space
        self.w = eigen_vc[:, :self.fda_components]
        return self

    # Transform the data using first PCA anc then the FDA
    def transform(self, X):
        X = self.pca.transform(X)
        return np.matmul(X, self.w)

    # Condenses the fit and the transform function together
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _predict_element(self, x):
        # Just for testing purpose on a sigle element of data
        x_t = np.matmul(x, self.w)

        self.m_class_t = np.matmul(self.m_class, self.w)

        mean_dist = []
        for i in range(self.fda_components + 1):
            mean_dist.append(np.linalg.norm(self.m_class_t[i] - x_t))

        return np.argmin(mean_dist), mean_dist

    # Returns an array of Labels that the FDA predicted
    def predict(self, X):
        # X data to predict

        # transforms the data
        X_t = self.transform(X)
        self.m_class_t = np.matmul(self.m_class, self.w)

        # Calculate the prediction usigng the distances from the classes means
        y_pred = []

        for i in range(X_t.shape[0]):
            mean_dist = []
            for j in range(self.fda_components + 1):
                mean_dist.append(np.linalg.norm(self.m_class_t[j] - X_t[i]))
            y_pred.append(np.argmin(mean_dist))

        return np.asarray(y_pred)

    # Necessary function to be compatible with GridSearchCV of sklearn interface
    def score(self, X, y):
        if not X.shape[0] == y.shape[0]:
            print('Error: Data and Targets are of incongruent dimentions')
        y_p = self.predict(X)
        return accuracy_score(y, y_p)

    # Necessary function to be compatible with GridSearchCV of sklearn interface
    def get_params(self, deep=False):
        dic = {
            'fda_components': self.fda_components,
            'kernel': self.kernel,
            'pca_components': self.pca_components,
            'k_degree': self.k_degree,
            'k_gamma': self.k_gamma,
            'k_coef0': self.k_coef0}
        return dic

    # Necessary function to be compatible with GridSearchCV of sklearn interface
    def set_params(
        self,
        pca_components=None,
        fda_components=None,
        kernel='linear',
        k_degree=3,
        k_gamma=None,
        k_coef0=1
    ):
        self.pca_components = pca_components
        self.fda_components = fda_components
        self.kernel = kernel
        self.k_degree = k_degree
        self.k_gamma = k_gamma
        self.k_coef0 = k_coef0
        return self

class FDA:
    # Fischer Discriminant Analysis
    def __init__(
        self,
        fda_components=None
    ):

        self.fda_components = fda_components
        return

    def _mean_class(self, X_class):

        mean = np.mean(np.concatenate(X_class, 0), 0)
        mean_class = np.array([np.mean(xc, 0) for xc in X_class])
        return mean_class, mean

    def _scatter_within(self, X_class, mean_class):
        n_features = mean_class.shape[1]
        S_w = np.zeros((n_features, n_features))

        for xc, mc in zip(X_class, mean_class):
            tmp = xc - mc[np.newaxis, :]
            S_w += np.matmul(tmp.T, tmp)

        return S_w

    def _scatter_total(self, X, mean):
        tmp = X - mean[np.newaxis, :]
        S_t = np.matmul(tmp.T, tmp)
        return S_t

    def _sparability(self, eigen_vc, eigen_vl, S_b, S_w):
        w = np.expand_dims(eigen_vc.T, 1)
        a = np.squeeze(np.matmul(np.matmul(w, S_b),
                       np.transpose(w, [0, 2, 1])), (1, 2))
        b = np.squeeze(np.matmul(np.matmul(w, S_w),
                       np.transpose(w, [0, 2, 1])), (1, 2))

        idx_nonzero = np.array(np.nonzero(b))[0]
        a = a[idx_nonzero]
        b = b[idx_nonzero]
        sep = a / b
        eigen_vl, eigen_vc = eigen_vl[idx_nonzero], eigen_vc[:, idx_nonzero]
        idx = sep.argsort()[::-1]

        return eigen_vl[idx], eigen_vc[:, idx]

    def fit(self, X, y, **params):
        # X Dataset
        # y labels
        # Accepts dictionary to conform to sklearn class so that its compatible with GridSearchCV

        # All unique labels
        unique_y = np.unique(y)

        # setting the number of components at the maximum possible if None value is set
        if self.fda_components == None:
            self.fda_components = np.min([X.shape[1], len(unique_y)-1])

        # Rearrage the data in class order
        X_class = [X[y == y_class] for y_class in unique_y]
        # Calculate the total mean and the classes mean
        self.m_class, self.mean = self._mean_class(X_class)
        # Calculate the total Scatter matrix
        self.S_t = self._scatter_total(X, self.mean)
        # Calculate of the scatter within matrix
        self.S_w = self._scatter_within(X_class, self.m_class)

        # Calculate the Scatter Between matrix using the simple subtraction
        self.S_b = self.S_t - self.S_w

        # Calculate the psudo_invers of the scatter within matrix
        self.i_S_w = np.linalg.pinv(self.S_w)

        sigma = np.matmul(self.i_S_w, self.S_b)

        # Eigenvalues and eigenvectors solution
        eigen_vl, eigen_vc = np.linalg.eig(sigma)

        # cosa fa?
        eigen_vl, eigen_vc = self._sparability(eigen_vc, eigen_vl, self.S_b, self.S_w)

        # Transformation Matrix that projects in the new space
        self.w = eigen_vc[:, :self.fda_components]
        return self

    # Transform the data using first PCA anc then the FDA
    def transform(self, X):
        return np.matmul(X, self.w)

    # Condenses the fit and the transform function together
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _predict_element(self, x):
        # Just for testing purpose on a sigle element of data
        x_t = np.matmul(x, self.w)

        self.m_class_t = np.matmul(self.m_class, self.w)

        mean_dist = []
        for i in range(self.fda_components + 1):
            mean_dist.append(np.linalg.norm(self.m_class_t[i] - x_t))

        return np.argmin(mean_dist), mean_dist

    # Returns an array of Labels that the FDA predicted
    def predict(self, X):
        # X data to predict

        # transforms the data
        X_t = self.transform(X)
        self.m_class_t = np.matmul(self.m_class, self.w)

        # Calculate the prediction usigng the distances from the classes means
        y_pred = []

        for i in range(X_t.shape[0]):
            mean_dist = []
            for j in range(self.fda_components + 1):
                mean_dist.append(np.linalg.norm(self.m_class_t[j] - X_t[i]))
            y_pred.append(np.argmin(mean_dist))

        return np.asarray(y_pred)

    # Necessary function to be compatible with GridSearchCV of sklearn interface
    def score(self, X, y):
        if not X.shape[0] == y.shape[0]:
            print('Error: Data and Targets are of incongruent dimentions')
        y_p = self.predict(X)
        return accuracy_score(y, y_p)

    # Necessary function to be compatible with GridSearchCV of sklearn interface
    def get_params(self, deep=False):
        dic = {
            'fda_components': self.fda_components}
        return dic

    # Necessary function to be compatible with GridSearchCV of sklearn interface
    def set_params(
        self,
        fda_components=None
    ):
        self.fda_components = fda_components
        return self
