import skfuzzy as fuzz
import numpy as np

class Fuzzy:
    def __init__(self,fs_num=3, ys = None, FSS = False):
        if fs_num < 2:
          print("Cf Error")
        self.triangular = fuzz.trimf
        self.trapezoidal = fuzz.trapmf
        self.num = fs_num
        self.FSS = FSS
        if self.FSS == False:
            self.min = np.min(ys)
            self.max = np.max(ys)
            self.p5 = np.percentile(ys, 5)
            self.p95 = np.percentile(ys, 95)
            self.triband = [np.percentile(ys,int(i)) for i in np.arange(5,95,(95 - 5)/(self.num - 2 + 1))]
        else:
            self.min = 0
            self.max = np.max(ys)
            self.p5 = (self.max - self.min) * 0.05 + self.min
            self.p95 = (self.max - self.min) * 0.95 + self.min
            self.triband = np.arange(self.p5, self.p95, (self.p95 - self.p5) / (self.num - 2 + 1))

    def get_membership(self,x):
        # The input x is a one-dimensional tensor.
        x = np.squeeze(x,axis=0)
        p_5 = self.p5
        p_95 = self.p95

        memmbership = np.zeros((self.num,len(x)))
        if self.triband[-1] >= p_95:
            self.triband[-1] = p_95
        memmbership[0] = self.trapezoidal(x,[self.min,self.min,p_5,self.triband[1]])
        memmbership[-1] = self.trapezoidal(x,[self.triband[-1],p_95,self.max,self.max])

        for i in range(self.num - 2):
            if i == 0:
                if len(self.triband)<3:
                    memmbership[i+1] = self.triangular(x,[p_5,self.triband[1],p_95])
                else:
                    memmbership[i+1] = self.triangular(x,[p_5,self.triband[1],self.triband[2]])
            elif i == self.num -3:
                memmbership[i + 1] = self.triangular(x, [self.triband[-2], self.triband[-1],p_95])
            else :
                memmbership[i+1] = self.triangular(x, [self.triband[i],self.triband[i+1],self.triband[i+2]])
        return memmbership


