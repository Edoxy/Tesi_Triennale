import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator


class Fda_kpca:
    def __init__(
        self,
        *,
        pca_components=None,
        max_components=None,
        kernel='linear',
        k_degree=3,
        k_gamma=None,
        k_coef0=1
    ):
        self.max_components = max_components
        self.pca_components = pca_components
        self.kernel = kernel
        self.k_degree = k_degree
        self.k_gamma = k_gamma
        self.k_coef0=k_coef0

    def _calc_mean_cls(self, X_cls):
        m = np.mean(np.concatenate(X_cls, 0), 0)
        m_cls = np.array([np.mean(xc, 0) for xc in X_cls])

        return m_cls, m

    def _calc_scatter_between(self, m_cls, m, X_cls):
        S_b = np.zeros((len(m), len(m)))
        for i, mc in enumerate(m_cls):
            tmp = (mc - m)[np.newaxis, :]
            S_b += (np.matmul(tmp.T, tmp) * len(X_cls[i]))

        return S_b

    def _calc_scatter_within(self, X_cls, m_cls):
        n_features = m_cls.shape[1]
        S_w = np.zeros((n_features, n_features))

        for xc, mc in zip(X_cls, m_cls):
            tmp = xc - mc[np.newaxis, :]
            S_w += np.matmul(tmp.T, tmp)

        return S_w

    def _calc_scatter_total(self, X, m):
        tmp = X - m[np.newaxis, :]
        S_t = np.matmul(tmp.T, tmp)
        return S_t

    def _get_evc_separability(self, evc, evl, S_b, S_w):
        w = np.expand_dims(evc.T, 1)
        a = np.squeeze(np.matmul(np.matmul(w, S_b),
                       np.transpose(w, [0, 2, 1])), (1, 2))
        b = np.squeeze(np.matmul(np.matmul(w, S_w),
                       np.transpose(w, [0, 2, 1])), (1, 2))

        idx_nonzero = np.array(np.nonzero(b))[0]
        a = a[idx_nonzero]
        b = b[idx_nonzero]
        sep = a / b
        evl, evc = evl[idx_nonzero], evc[:, idx_nonzero]
        idx = sep.argsort()[::-1]

        return evl[idx], evc[:, idx]

# Function that fits the model to the data
    def fit(self, X, y, **params):
        #X: dataset
        # y: data classes

        # array containing all the unique data classes
        uq_y = np.unique(y)

        # pca
        self.n_c_pca = X.shape[0] - uq_y.shape[0]
        self.pca = KernelPCA(n_components=self.n_c_pca,
            kernel=self.kernel,
            degree=self.k_degree,
            gamma=self.k_gamma,
            coef0=self.k_coef0)
        X = self.pca.fit_transform(X)

        # setting the number of components at the mazimìmum possible if not
        if not self.max_components:
            self.max_components = np.min(
                [X.shape[1], len(uq_y), self.max_components])

        # cosa fa? separa in classi forse
        X_cls = [X[y == uy] for uy in uq_y]

        # separa info sulle medie delle classi
        self.m_cls, self.m = self._calc_mean_cls(X_cls)
        # calcolo della scatter within
        self.S_w = self._calc_scatter_within(X_cls, self.m_cls)
        # calcolo dello scatter totale
        self.S_t = self._calc_scatter_total(X, self.m)

        # S_b = self._calc_scatter_between(self.m_cls, self.m, X_cls)
        self.S_b = self.S_t - self.S_w

        # pordotto matriciale
        Sigma = np.matmul(np.linalg.pinv(self.S_w), self.S_b)

        # risoluzione di autovettori e autovalori
        evl, evc = np.linalg.eig(Sigma)
        # evc, _ = svd_flip(evc, np.zeros_like(evc).T)

        # cosa fa?
        evl, evc = self._get_evc_separability(evc, evl, self.S_b, self.S_w)

        # matrice che trasforma e proietta nel nuovo spazio
        self.w = evc[:, :self.max_components]
        return self

    def transform(self, X):
        X = self.pca.transform(X)
        return np.matmul(X, self.w)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict_element(self, x):
        x_t = np.matmul(x, self.w)

        self.m_cls_t = np.matmul(self.m_cls, self.w)

        dist_medie = []
        for i in range(self.max_components + 1):
            dist_medie.append(np.linalg.norm(self.m_cls_t[i] - x_t))

        return np.argmin(dist_medie)

    def predict(self, X):
        X_t = self.transform(X)

        self.m_cls_t = np.matmul(self.m_cls, self.w)

        y_pred = []

        for i in range(X_t.shape[0]):
            mean_dist = []
            for j in range(self.max_components + 1):
                mean_dist.append(np.linalg.norm(self.m_cls_t[j] - X_t[i]))
            y_pred.append(np.argmin(mean_dist))

        return np.asarray(y_pred)

    def score(self, X, y):
        if not X.shape[0] == y.shape[0]:
            print('Error: Data and Targets are of incongruent dimentions')
        y_p = self.predict(X)
        return accuracy_score(y, y_p)

    def get_params(self, deep = False):
        dic = {
        'max_components': self.max_components,
        'kernel': self.kernel,
        'pca_components': self.pca_components,
        'k_degree': self.k_degree,
        'k_gamma': self.k_gamma,
        'k_coef0': self.k_coef0}
        return dic
    
    def set_params(
        self,
        pca_components=None,
        max_components=None,
        kernel='linear',
        k_degree=3,
        k_gamma=None,
        k_coef0=1
    ):
        self.pca_components = pca_components
        self.max_components = max_components
        self.kernel = kernel
        self.k_degree = k_degree
        self.k_gamma = k_gamma
        self.k_coef0 = k_coef0
        return self