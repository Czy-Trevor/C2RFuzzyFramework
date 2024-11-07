'''
JDAR, JPDAR AND SDAR
based on SEED-VIG
'''
from tqdm import tqdm
import time
import JPDAR
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics
from joblib import Parallel, delayed
from fuzzy import Fuzzy

np.random.seed(0)
loaded_x = np.load('/mnt/data/zycui/SEED/SEEDX_db_0norm.npz')
loaded_y = np.load('/mnt/data/zycui/SEED/SEEDY_db_0norm.npz')
X = [loaded_x[key] for key in loaded_x.files]
Y = [loaded_y[key] for key in loaded_y.files]

def target_train(num, T, SDAR, KR, dim, max_fs, X, Y):
    target = num
    Y_tar = []
    for_starttime = time.time()
    Xt_ori = X[target]
    Yt = Y[target].reshape(-1, 1)
    Ysour_all = Y.copy()
    del Ysour_all[target]
    Ys_numpy = np.empty((0, 1))
    for i in range(len(Ysour_all)):
        Ys_numpy = np.vstack((Ys_numpy, Ysour_all[i]))

    for sour in tqdm(range(len(X)), desc="Processing", unit="soursub"):
        if sour != target:
            Xs_ori = X[sour]
            Ys = Y[sour].reshape(-1, 1)
            Ysweight = np.zeros([max_fs + 1, 1])
            for i in range(3, max_fs + 1):
                Fuzzy_ada = Fuzzy(fs_num=i, ys=Ys_numpy, FSS=True)
                Ys_membership = Fuzzy_ada.get_membership(Ys.T)
                max_indices = np.argmax(Ys_membership, axis=0)
                temp_matrix = np.copy(Ys_membership)
                temp_matrix[max_indices, np.arange(temp_matrix.shape[1])] = -np.inf
                second_max_indices = np.argmax(temp_matrix, axis=0)

                max_values = Ys_membership[max_indices, np.arange(Ys_membership.shape[1])]
                second_max_values = Ys_membership[second_max_indices, np.arange(Ys_membership.shape[1])]
                diff = max_values - second_max_values
                tem = np.mean(diff)
                Ysweight[i] = tem

            ind = np.argmax(Ysweight)
            print(f"FSS: source = {sour}, ind = {ind}")
            JPDAF = JPDAR.JPDAR(fs_num=ind, T=T, dim=dim, mmd_type="djp-mmd", ys=Ys_numpy, FSS=True)
            Xs_new, Xt_new, ytar = JPDAF.fit(Xs_ori, Ys, Xt_ori, Yt, SDAR=SDAR, KR_=KR)

            model = KNeighborsRegressor(n_neighbors=1)
            model.fit(Xs_new, Ys.ravel())
            Y_tar_predict = model.predict(Xt_new)
            Y_tar_predict[Y_tar_predict > 1] = 1
            Y_tar_predict[Y_tar_predict < 0] = 0
            Y_tar.append(Y_tar_predict)

    Y_aver = sum(Y_tar) / len(Y_tar)
    Y_aver = Y_aver.reshape(-1, 1)
    mse = sklearn.metrics.mean_squared_error(Yt, Y_aver)
    pearson_corr_coeff = np.corrcoef(Yt.reshape(1, -1), Y_aver.reshape(1, -1))[0, 1]

    for_endtime = time.time()
    running_time = for_endtime - for_starttime
    print(f"Subject {target + 1}")
    print(f"Source time: {running_time:.3f} s")
    print(np.sqrt(mse))
    print(pearson_corr_coeff)
    return np.sqrt(mse), pearson_corr_coeff


def train_all(T=10, SDAR=True, KR=1, dim=17, max_fs=10):
    sub_num = len(X)
    start_time = time.time()

    result = Parallel(n_jobs=3)(delayed(target_train)(i, T, SDAR, KR, dim, max_fs, X, Y) for i in range(sub_num))
    mselist = [result[i][0] for i in range(len(result))]
    CClist = [result[i][1] for i in range(len(result))]

    end_time = time.time()
    code_time = end_time - start_time
    print(f"Time: {code_time:.5f} s")
    print('RMSELoss\n')
    for item in mselist:
        print(item)
    print('CC\n')
    for item in CClist:
        print(item)
    print('\n')
    mse_array = np.array(mselist)
    CC_array = np.array(CClist)

    return np.mean(mse_array), np.mean(CC_array)


if __name__ == '__main__':
    mse, cc = train_all(T=10,SDAR = True, max_fs=10, KR=1)
    print("FSS-SDAR")
    print(f"max_fs = {10},avemse = {mse},avecc = {cc}")



