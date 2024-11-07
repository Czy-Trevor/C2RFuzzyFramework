import skfuzzy as fuzz
import numpy as np


class Fuzzy:
    def __init__(self,fs_num=3):
        if fs_num < 2:
          print("only 1 fuzzy")
        self.triangular = fuzz.trimf
        self.trapezoidal = fuzz.trapmf
        self.num = fs_num
    def get_membership(self,x):
        p_5 = np.percentile(x, 5)
        p_95 = np.percentile(x, 95)
        triband = [np.percentile(x, int(i)) for i in np.arange(5, 95, (95 - 5) / (self.num - 2 + 1))]
        min_ = np.min(x)
        max_ = np.max(x)
        x = np.squeeze(x)
        memmbership = np.zeros((self.num,len(x)))
        if triband[-1] >= p_95:
            triband[-1] = p_95
        if min_ <= min_ and min_ <= p_5 and p_5 <= triband[1]:
            memmbership[0] = self.trapezoidal(x,[min_,min_,p_5,triband[1]])
            memmbership[-1] = self.trapezoidal(x,[triband[-1],p_95,max_,max_])
        else:
            print(f"min = {min_}")
            print(f"p_5 = {p_5}")
            print(f"triband = {triband}")
            raise Exception("Possible Loss Error")
        for i in range(self.num - 2):
            if i == 0:
                if len(triband)<3:
                    memmbership[i+1] = self.triangular(x,[p_5,triband[1],p_95])
                else:
                    memmbership[i+1] = self.triangular(x,[p_5,triband[1],triband[2]])
            elif i == self.num -3:
                memmbership[i + 1] = self.triangular(x, [triband[-2], triband[-1],p_95])
            else :
                memmbership[i+1] = self.triangular(x, [triband[i],triband[i+1],triband[i+2]])
        return memmbership
