'''
JDAR, JPDAR AND SDAR
based on driving_data
'''
from tqdm import tqdm
import time
import JDAR
import JPDAR
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics
from joblib import Parallel, delayed

np.random.seed(0)
loaded_x = np.load('/mnt/data/zycui/DATA/X_list.npz')
loaded_y = np.load('/mnt/data/zycui/DATA/Y_list.npz')
X = [loaded_x[key] for key in loaded_x.files]
Y = [loaded_y[key] for key in loaded_y.files]

def target_train(num, fsnum, T, type, JDAmu, JPDAmu, SDAR, KR, dim, X, Y):
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
            JDAF = JDAR.JDAR(fs_num=fsnum, T=T, dim=dim, mu=JDAmu, ys=Ys_numpy)
            JPDAF = JPDAR.JPDAR(fs_num=fsnum, T=T, dim=dim, mmd_type="djp-mmd", ys=Ys_numpy, mu=JPDAmu)
            if type == 'JDA':
                Xs_new, Xt_new, yt = JDAF.fit(Xs_ori, Ys, Xt_ori, Yt, KR_=KR)
            elif type == 'JPDA':
                Xs_new, Xt_new, yt = JPDAF.fit(Xs_ori, Ys, Xt_ori, Yt, SDAR=SDAR, KR_=KR)

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
    print(f"{running_time:.3f} s")
    print(np.sqrt(mse))
    print(pearson_corr_coeff)
    return np.sqrt(mse), pearson_corr_coeff

def train_all(fsnum, T=10, type='JDA', JDAmu=0.5, SDAR=True, KR=1, dim=30, JPDAmu=0.1):
    print(f"T = {T}, type = {type}, SDAR = {SDAR}")
    sub_num = len(X)
    start_time = time.time()
    Y_averall = []

    for num in range(sub_num):
        mse, pearson_corr_coeff = target_train(num, fsnum, T, type, JDAmu, JPDAmu, SDAR, KR, dim, X, Y)
        Y_averall.append((mse, pearson_corr_coeff))

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.3f} s")
    return Y_averall

if __name__ == '__main__':
    rmse, cc = train_all(3, T=10, type='JDA', SDAR = False)
    print("JDAR")
    print(f"fs = {3},avermse = {rmse},avecc = {cc}")

    rmse, cc = train_all(3, T=10, type='JPDA', SDAR = False)
    print("JPDAR")
    print(f"fs = {3},avermse = {rmse},avecc = {cc}")

    rmse, cc = train_all(3, T=10, type='JPDA',SDAR = True)
    print("SDAR")
    print(f"fs = {3},avermse = {rmse},avecc = {cc}")



