import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_kernels

# ----- python fast kernel matrix:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
# https://stackoverflow.com/questions/7391779/fast-kernel-matrix-computation-python
# https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
# https://stackoverflow.com/questions/36324561/fast-way-to-calculate-kernel-matrix-python?rq=1

# ----- python fast scatter matrix:
# https://stackoverflow.com/questions/31145918/fast-weighted-scatter-matrix-calculation

class My_kernel_FDA:

    def __init__(self, n_components=None, kernel=None):
        self.n_components = n_components
        self.Theta = None
        self.X_train = None
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = 'linear'

    def fit_transform(self, X, y):
        # X: columns are sample, rows are features
        self.fit(X=X, y=y)
        X_transformed = self.transform(X=X, y=y)
        return X_transformed

    def fit(self, X, y):
        # X: columns are sample, rows are features
        self.X_train = X
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n_samples = X.shape[1]
        labels_of_classes = list(set(y.ravel()))
        n_classes = len(labels_of_classes)
        # ------ M_*:
        Kernel_allSamples_allSamples = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        M_star = Kernel_allSamples_allSamples.sum(axis=1)
        M_star = M_star.reshape((-1,1))
        M_star = (1 / n_samples) * M_star
        # ------ M_c and M:
        M = np.zeros((n_samples, n_samples))
        for class_index in range(n_classes):
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            # ------ M_c:
            Kernel_allSamples_classSamples = pairwise_kernels(X=X.T, Y=X_class.T, metric=self.kernel)
            M_c = Kernel_allSamples_classSamples.sum(axis=1)
            M_c = M_c.reshape((-1, 1))
            M_c = (1 / n_samples_of_class) * M_c
            # ------ M:
            M = M + n_samples_of_class * (M_c - M_star).dot((M_c - M_star).T)
        # ------ N:
        N = np.zeros((n_samples, n_samples))
        for class_index in range(n_classes):
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            Kernel_allSamples_classSamples = pairwise_kernels(X=X.T, Y=X_class.T, metric=self.kernel)
            K_c = Kernel_allSamples_classSamples
            H_c = np.eye(n_samples_of_class) - (1 / n_samples_of_class) * np.ones((n_samples_of_class, n_samples_of_class))
            N = N + K_c.dot(H_c).dot(K_c.T)
        # ------ kernel Fisher directions:
        epsilon = 0.00001  #--> to prevent singularity of matrix N
        eig_val, eig_vec = LA.eigh(inv(N + epsilon*np.eye(N.shape[0])).dot(M))
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            Theta = eig_vec[:, :self.n_components]
        else:
            Theta = eig_vec[:, :n_classes-1]
        self.Theta = Theta

    def transform(self, X, y):
        # X: columns are sample, rows are features
        # X_transformed: columns are sample, rows are features
        Kernel_train_input = pairwise_kernels(X=self.X_train.T, Y=X.T, metric=self.kernel)
        X_transformed = (self.Theta.T).dot(Kernel_train_input)
        return X_transformed

    def transform_outOfSample_all_together(self, X, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        Kernel_train_input = pairwise_kernels(X=self.X_train.T, Y=X.T, metric=self.kernel)
        X_transformed = (self.Theta.T).dot(Kernel_train_input)
        return X_transformed

    def _build_kernel_matrix(self, X, kernel_func, option_kernel_func=None):  # --> K = self._build_kernel_matrix(X=X, kernel_func=self._radial_basis)
        # https://stats.stackexchange.com/questions/243104/how-to-build-and-use-the-kernel-trick-manually-in-python
        # X = X.T
        n_samples = X.shape[1]
        n_features = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            xi = X[:, i]
            for j in range(n_samples):
                xj = X[:, j]
                K[i, j] = kernel_func(xi, xj, option_kernel_func)
        return K

    def _radial_basis(self, xi, xj, gamma=None):
        if gamma is None:
            n_features = xi.shape[0]
            gamma = 1 / n_features
        r = (np.exp(-gamma * (LA.norm(xi - xj) ** 2)))
        return r

    def _separate_samples_of_classes(self, X, y):
        # X --> rows: features, columns: samples
        # X_separated_classes --> rows: features, columns: samples
        X = X.T
        y = np.asarray(y)
        y = y.reshape((-1, 1))
        yX = np.column_stack((y, X))
        yX = yX[yX[:, 0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
        y = yX[:, 0]
        X = yX[:, 1:]
        labels_of_classes = list(set(y))
        number_of_classes = len(labels_of_classes)
        dimension_of_data = X.shape[1]
        X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        class_index = 0
        index_start_new_class = 0
        n_samples = X.shape[0]
        for sample_index in range(1, n_samples):
            if y[sample_index] != y[sample_index - 1] or sample_index == n_samples-1:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
                index_start_new_class = sample_index
                class_index = class_index + 1
        for class_index in range(number_of_classes):
            X_class = X_separated_classes[class_index]
            X_separated_classes[class_index] = X_class.T
        return X_separated_classes