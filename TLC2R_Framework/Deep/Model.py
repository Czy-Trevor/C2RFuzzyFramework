
import torch
import torch.nn as nn
import lmmd
import fuzzy
import torch.nn.functional as F
import mmd
import CORAL
import GRAM

class Model(nn.Module):
    def __init__(self, jmmd_loss,dim_input = 60, num_fs = 3,num_hidden=1,MMD = 'DSAN'):
        super(Model, self).__init__()
        self.jmmd_loss = jmmd_loss
        self.feature_layers = nn.Sequential(nn.Linear(dim_input,dim_input),nn.ReLU())
        self.num_fs = num_fs
        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_fs)
        self.MMD = MMD
        self.num_hidden = num_hidden
        self.fuzzy_ = fuzzy.Fuzzy(fs_num=self.num_fs)
        if num_hidden!=0:
            self.bottle = nn.Sequential()
            for i in range(num_hidden):
                self.bottle.append(nn.Linear(dim_input,dim_input))
                self.bottle.append(nn.ReLU())
            self.cls_fc = nn.Linear(dim_input, 1)

        else:
            self.cls_fc = nn.Linear(dim_input, 1)

        self.l2v = nn.Sequential(nn.Linear(1, self.num_fs), nn.ReLU())
        self.l2v.append(nn.Linear(self.num_fs,  self.num_fs))
        self.l2v.append(nn.ReLU())

    def label2vector(self,label):
        vector = self.l2v(label)
        return F.softmax(vector)

    def forward(self, source, target,s_label):
        x = torch.cat((source,target),dim = 0)
        xf = self.feature_layers(x)
        if self.num_hidden != 0:
            xfb = self.bottle(xf)
        xpred = self.cls_fc(xfb)
        s_pred, t_label = xpred.chunk(2, dim=0)
        source_fb, target_fb = xfb.chunk(2, dim=0)
        target_f = self.feature_layers(target)

        if self.num_hidden != 0:
            target_fb = self.bottle(target_f)
        t_label = self.cls_fc(target_fb)

        if self.MMD == 'DAN':
            loss_tran = mmd.mmd_rbf_accelerate(source_fb, target_fb)

        elif self.MMD =='CORAL':
            loss_tran = CORAL.CORAL(source_fb, target_fb)

        elif self.MMD =='JANR':
            sys = s_label.cpu().data.numpy()
            tys = t_label.cpu().data.numpy()
            s_memmbership = self.fuzzy_.get_membership(sys).transpose()
            t_memmbership = self.fuzzy_.get_membership(tys).transpose()
            s_memmbership = torch.from_numpy(s_memmbership).cuda().float()
            t_memmbership = torch.from_numpy(t_memmbership).cuda().float()
            loss_tran = self.jmmd_loss((source_fb,s_memmbership),(target_fb,t_memmbership))

        elif self.MMD == 'GRAM':
            loss_tran = GRAM.DARE_GRAM_LOSS(source_fb, target_fb,target_fb.device)

        elif self.MMD == 'DSANR':
            sys = s_label.cpu().data.numpy()
            tys = t_label.cpu().data.numpy()
            s_memmbership = self.fuzzy_.get_membership(sys).transpose()
            t_memmbership = self.fuzzy_.get_membership(tys).transpose()
            loss_tran = self.lmmd_loss.get_loss(source_fb, target_fb, s_memmbership, t_memmbership)

        elif self.MMD == 'DANN':
            #caculate outside
            loss_tran = 0
        else:
            raise Exception("MMD type ERROR")

        return s_pred, loss_tran, source_fb, target_fb


    def predict(self, x):
        x = self.feature_layers(x)
        if self.num_hidden!=0:
            x = self.bottle(x)
        return self.cls_fc(x)
