# -*- coding: utf-8 -*-
# @Time    : 2021/7/10 16:01
# @Author  : Wen Zhang
# @File    : JPDA_compare_python.py
# Reference: https://github.com/jindongwang/transferlearning/blob/master/code/traditional/JDA

import numpy as np
import scipy.io
import scipy.linalg
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
import scipy.io
import scipy.linalg
from sklearn.neighbors import KNeighborsRegressor
from fuzzy import Fuzzy

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


def get_matrix_M(Ys_membership, Y_tar_pseudo, ns, nt, C, mu,fuzzy, type='djp-mmd'):


    if type == 'jp-mmd':
        Ns = 1 / ns * Ys_membership.T
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Y_tar_pseudo = Y_tar_pseudo.reshape(-1, 1)
            Yt_membership = fuzzy.get_membership(Y_tar_pseudo.T)
            Nt = 1 / nt * Yt_membership.T
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        M = Rmin / np.linalg.norm(Rmin, 'fro')

    if type == 'djp-mmd':
        # For transferability
        ysm = Ys_membership.sum()
        Ns = 1 / ysm * Ys_membership.T
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Y_tar_pseudo = Y_tar_pseudo.reshape(-1, 1)
            Yt_membership = fuzzy.get_membership(Y_tar_pseudo.T)
            ytm = Yt_membership.sum()
            Nt = 1 / ytm * Yt_membership.T
            '''
            Nt:ndarray , num_sanples * num_fs
            '''
            inds = np.where(np.sum(Ns,axis=0) == 0)
            indt = np.where(np.sum(Nt,axis=0) == 0)
            Ns[:,inds] = 0
            Nt[:, indt] = 0
            Ns[:,indt] = 0
            Nt[:, inds] = 0

        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        Rmin = Rmin / np.linalg.norm(Rmin, 'fro')

        # For discriminability
        Ms = np.zeros([ns, (C - 1) * C])
        Mt = np.zeros([nt, (C - 1) * C])
        for i in range(C):
            idx = np.arange((C - 1) * i, (C - 1) * (i + 1))
            Ms[:, idx] = np.tile(Ns[:, i], (C - 1, 1)).T
            tmp = np.arange(C)
            Mt[:, idx] = Nt[:, tmp[tmp != i]]
        Rmax = np.r_[np.c_[np.dot(Ms, Ms.T), np.dot(-Ms, Mt.T)], np.c_[np.dot(-Mt, Ms.T), np.dot(Mt, Mt.T)]]
        Rmax = Rmax / np.linalg.norm(Rmax, 'fro')
        M = Rmin - mu * Rmax

    return M


def get_matrix_M_SDAR(Ys_membership, Y_tar_pseudo, ns, nt, C, mu,fuzzy, type='djp-mmd'):

    M = 0
    if type == 'jp-mmd':
        # For transferability
        Ns = 1 / ns * Ys_membership.T
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Y_tar_pseudo = Y_tar_pseudo.reshape(-1, 1)
            Yt_membership = fuzzy.get_membership(Y_tar_pseudo.T)
            Nt = 1 / nt * Yt_membership.T
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        M = Rmin / np.linalg.norm(Rmin, 'fro')

    if type == 'djp-mmd':
        # For transferability
        zero_flag = np.ones((C))
        Ns = 1 / ns * Ys_membership.T
        Nt = np.zeros([nt, C])
        sour_weight = Ys_membership.sum(axis=1)
        for i in range(C):
            if sour_weight[i] != 0:
                Ns[:, i] = 1 / sour_weight[i] * Ys_membership[i].T
            else:
                Ns[:, i] = 0
                zero_flag[i] = 0

        if Y_tar_pseudo is not None:
            Y_tar_pseudo = Y_tar_pseudo.reshape(-1, 1)
            Yt_membership = fuzzy.get_membership(Y_tar_pseudo.T)
            tar_weight = Yt_membership.sum(axis=1)
            for i in range(C):
                if tar_weight[i] != 0:
                    Nt[:, i] = 1 / tar_weight[i] * Yt_membership[i].T
                else:
                    Nt[:, i] = 0
                    zero_flag[i] = 0

            for i in range(C):
                if zero_flag[i] ==0:
                    Nt[:, i] = 0
                    Ns[:, i] = 0
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        Rmin = Rmin / np.linalg.norm(Rmin, 'fro')

        # For discriminability
        Ms = np.zeros([ns, (C - 1) * C])
        Mt = np.zeros([nt, (C - 1) * C])
        for i in range(C):
            idx = np.arange((C - 1) * i, (C - 1) * (i + 1))
            Ms[:, idx] = np.tile(Ns[:, i], (C - 1, 1)).T
            tmp = np.arange(C)
            Mt[:, idx] = Nt[:, tmp[tmp != i]]
        Rmax = np.r_[np.c_[np.dot(Ms, Ms.T), np.dot(-Ms, Mt.T)], np.c_[np.dot(-Mt, Ms.T), np.dot(Mt, Mt.T)]]
        Rmax = Rmax / np.linalg.norm(Rmax, 'fro')
        M = Rmin - mu * Rmax

    return M

class JPDAR:
    def __init__(self, kernel_type='primal', mmd_type='djp-mmd', dim=30,fs_num=3, lamb=1, gamma=1, mu=0.1, T=10,ys = None, FSS= False):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.mmd_type = mmd_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.mu = mu
        self.fs_num = fs_num
        self.T = T
        self.fuzzy_set = Fuzzy(fs_num, ys = ys, FSS = FSS)

    def fit(self, Xs, Ys, Xt, Yt, KR_ = 1,SDAR = True):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        X = np.hstack((Xs.T, Xt.T))
        X = np.dot(X, np.diag(1. / np.linalg.norm(X, axis=0)))
        m, n = X.shape  # 800, 2081
        ns, nt = len(Xs), len(Xt)
        C = self.fs_num
        H = np.eye(n) - 1 / n * np.ones((n, n))

        Ys_membership = self.fuzzy_set.get_membership(Ys.T)

        Y_tar_pseudo = None
        for itr in range(self.T):
            if SDAR == True:
                M = get_matrix_M_SDAR(Ys_membership, Y_tar_pseudo, ns, nt, C, self.mu,self.fuzzy_set,type=self.mmd_type)
            else:
                M = get_matrix_M(Ys_membership, Y_tar_pseudo, ns, nt, C, self.mu,self.fuzzy_set,type=self.mmd_type)
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            model = KNeighborsRegressor(n_neighbors=KR_)
            model.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = model.predict(Xt_new)

        return Xs_new, Xt_new,Y_tar_pseudo.reshape(-1,1)

