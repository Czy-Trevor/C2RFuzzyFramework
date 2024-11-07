# Reference: https://github.com/jindongwang/transferlearning/blob/master/code/traditional/JDA
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsRegressor
from fuzzy import Fuzzy
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


class JDAR:
    # The combination of JDA (Joint Domain Adaptation) and fuzzy set,
    # used for aligning the conditional probability distribution in regression problems.
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10, fs_num=3 ,ys = None,mu = 0.5):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.mu = mu
        self.lamb = lamb
        self.gamma = gamma
        self.T = T
        self.fs_num = fs_num
        self.fuzzy_set = Fuzzy(fs_num, ys = ys)

    def fit(self, Xs, Ys, Xt, Yt,KR_ = 1):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        ###################################################
        fuzzy_C = self.fs_num
        M0 = e * e.T * fuzzy_C
        H = np.eye(n) - 1 / n * np.ones((n, n))
        ###################################################
        Ys_membership = self.fuzzy_set.get_membership(Ys.T)
        M = 0
        Y_tar_pseudo = None

        for t in range(self.T):
            N = 0

            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                Y_tar_pseudo = Y_tar_pseudo.reshape(-1,1)
                # Dataset interval restrictions (or range limits)
                Y_tar_pseudo[Y_tar_pseudo > 1] = 1
                Y_tar_pseudo[Y_tar_pseudo < 0] = 0
                ###################################################
                Yt_membership = self.fuzzy_set.get_membership(Y_tar_pseudo.T)
                sour_weight = Ys_membership.sum(axis= 1)
                tar_weight = Yt_membership.sum(axis = 1)

                for c in range(fuzzy_C):
                    e = np.zeros((n, 1))
                    if tar_weight[c] != 0:
                        e[ns:] = (-1 * Yt_membership[c]).T.reshape(-1, 1) / tar_weight[c]
                    else:
                        e[:] = 0
                    if sour_weight[c] != 0:
                        e[:ns]= Ys_membership[c].T.reshape(-1,1) / sour_weight[c]
                    else:
                        e[:] = 0

                    e[np.isinf(e)] = 0
                    e[np.isnan(e)] = 0
                    if c == 0:
                        N = N + np.dot(e, e.T)
                    elif c == fuzzy_C:
                        N = N + np.dot(e, e.T)
                    else:
                        N = N + np.dot(e, e.T)

            M = (1-self.mu)*M0 + self.mu*N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            Xs_new = np.real(Xs_new)
            Xt_new = np.real(Xt_new)

            model = KNeighborsRegressor(n_neighbors=KR_)
            model.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = model.predict(Xt_new)

        return Xs_new, Xt_new,Y_tar_pseudo.reshape(-1,1)




