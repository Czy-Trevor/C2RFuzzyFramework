import torch
# Reference: https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA/DSAN
import torch.nn as nn
import numpy as np

class LMMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))

        L2_distance = ((total0-total1)**2).sum(2)

        L2 = L2_distance.detach()
        s = source.detach()
        t = target.detach()

        L2_distance_numpy = L2.cpu().numpy()
        s_numpy = s.cpu().numpy()
        t_numpy = t.cpu().numpy()
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)


    def get_loss(self, source, target, s_label, t_label):
        '''

        :param source: tensor,batch * dim
        :param target: tensor,batch * dim
        :param s_label: ndarray,batch * C_f
        :param t_label: ndarray,batch * C_f
        :return:
        '''
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()
        all_zero_s = torch.allclose(source, torch.zeros_like(source))
        all_zero_t = torch.allclose(target, torch.zeros_like(target))
        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            kernels = self.guassian_kernel(source, target,
                                           kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            return loss

        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss


    def cal_weight(self, s_label, t_label):
        '''

        :param s_label:  num_fs*batch_size ,ndarray
        :param t_label:  num_fs*batch_size ,ndarray
        :return:
        '''
        s_sca_label = s_label
        s_vec_label = s_sca_label.copy()
        # Normalize sample dimension of membership values
        s_sum = np.sum(s_vec_label, axis=0,keepdims=True)
        # Avoid having no membership for a certain class within a batch
        inds = np.where(s_sum == 0)

        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label
        t_vec_label = t_sca_label.copy()
        t_sum = np.sum(t_vec_label, axis=0,keepdims=True)
        indt = np.where(t_sum == 0)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        t_vec_label[indt,:] = 0
        t_vec_label[inds, :] = 0
        s_vec_label[indt,:] = 0
        s_vec_label[inds, :] = 0

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')