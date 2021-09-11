import numpy as np

'''
蝙蝠算法-Bat Algorithm
'''


class BA(object):
    def __init__(self, d, N_p, N_gen, Qmin, Qmax, lower_bound, upper_bound, func):#Step1: 种群初始化，即蝙蝠以随机方式在D维空间中扩散分布一组初始解。最大脉冲音量A0,最大脉冲率R0, 搜索脉冲频率范围[fmin,fmax],音量的衰减系数α，搜索频率的增强系数γ，搜索精度ε或最大迭代次数iter_max
        self.d = d  # 搜索维度
        self.N_p = N_p  # 个体数
        self.N_gen = N_gen  # 迭代次数
        self.A = 1 + np.random.random(self.N_p)  # 响度
        self.r = np.random.random(self.N_p)  # 脉冲发射率
        self.Qmin = Qmin  # 最小频率
        self.Qmax = Qmax  # 最大频率
        self.lower_bound = lower_bound  # 搜索区间下限
        self.upper_bound = upper_bound  # 搜索区间上限
        self.func = func  # 目标函数
        self.alpha = 0.85
        self.gamma = 0.9
        self.r0 = self.r
        self.Lb = self.lower_bound * np.ones(self.d)
        self.Ub = self.upper_bound * np.ones(self.d)
        self.Q = np.zeros(self.N_p)  # 频率
        self.v = np.zeros((self.N_p, self.d))  # 速度
        self.sol = np.zeros((self.N_p, self.d))  # 种群
        self.fitness = np.zeros(self.N_p)  # 个体适应度
        self.best = np.zeros(self.d)  # 最好的solution
        self.fmin = 0.0  # 最小fitness
    # 初始化蝙蝠种群
    def init_bat(self):#。Step2: 随机初始化蝙蝠的位置xi,并根据适应度值得优劣寻找当前的最优解x*。
        for i1 in range(self.N_p):
            self.sol[i1, :] = self.Lb + (self.Ub - self.Lb) * np.random.uniform(0, 1, self.d)
            self.fitness[i1] = self.func(self.sol[i1, :])
        #print(self.sol)

        self.fmin = np.min(self.fitness)
        fmin_arg = np.argmin(self.fitness)
        self.best = self.sol[fmin_arg, :]
    # 越界检查
    def simplebounds(self, s, lb, ub):
        for j1 in range(self.d):
            if s[j1] < lb[j1]:
                s[j1] = lb[j1]
            if s[j1] > ub[j1]:
                s[j1] = ub[j1]
        return s
    # 迭代部分 数学模型，相关参数，优化方法，迭代次数理论依据
    def start_iter(self):
        S = np.zeros((self.N_p, self.d))
        self.init_bat()
        for step in range(self.N_gen):
            for i2 in range(self.N_p):#Step3: 蝙蝠的搜索脉冲频率、速度和位置更新。种群在进化过程中每一下公式进行变化:
                if i2 < self.N_p :

                    self.Q[i2] = self.Qmin + (self.Qmin - self.Qmax) * np.random.uniform(0, 1) #fi=fmin+(fmax-fmin)xβ (1)
                    self.v[i2, :] = self.v[i2, :] + (self.sol[i2, :] - self.best) * self.Q[i2] #vi^t=vi^(t-1)+(xi^t-x*)xfi (2)
                    print( (self.sol[i2, :] - self.best) * self.Q[i2])
                    S[i2, :] = self.sol[i2, :] + self.v[i2, :] #xi^t=xi^(t-1)+vi^(t) (3)
                else:
                    #黄金正玄
                    #print('Loading')
                    t=(np.sqrt(5)-1)/2
                    a1=-np.pi+(1-t)*2*np.pi
                    a2=-np.pi+t*2*np.pi
                    R1=np.random.uniform(0, 1)
                    R2=np.random.uniform(0, 1)
                    #self.Q[i2] = self.Qmin + (self.Qmin - self.Qmax) * np.random.uniform(0, 1)
                    self.v[i2, :] = self.v[i2, :] + (self.sol[i2, :] - self.best) * self.Q[i2]
                    #self.v[i2, :]=
                    S[i2, :] = self.sol[i2, :] *R1+R2*np.sin(R1)*np.abs(a1*self.best-a2*self.sol[i2, :])





                S[i2, :] = self.simplebounds(S[i2, :], self.Lb, self.Ub)  # 越界检查

                if np.random.random() > self.r[i2]: #Step4:生成均匀分布随机数rand,如果rand>r,则对当前最优解进行随机扰动，产生一个新的解，并对新的解进行越界处理。
                    S[i2, :] = self.best + 0.001 * np.random.randn(self.d)  # 此处没有实现乘以响度平均值
                    S[i2, :] = self.simplebounds(S[i2, :], self.Lb, self.Ub)  # 越界检查

                Fnew = self.func(S[i2, :])
                if (Fnew <= self.fitness[i2]) and (np.random.random() < self.A[i2]):#Step5:生成均匀分布随机数rand,如果rand<Ai且f(xi)<f(x*),则接受步骤4产生的新解
                    self.sol[i2, :] = S[i2, :]
                    self.fitness[i2] = Fnew
                    self.A[i2] = self.alpha * self.A[i2]  # 响度更新 Ai^(t+1)=αAi^(t) (4)
                    self.r[i2] = self.r0[i2] * (1 - np.exp(-1 * self.gamma * step))  # 脉冲发射率更新 #ri^(t+1)=R0[1-exp(-γt)] (5)

                if Fnew <= self.fmin:#Step6:对所有蝙蝠的适应度值进行排序,找出当前的最优解和最优值。取到最小值，更新best
                    self.best = S[i2, :]
                    self.fmin = Fnew
            print(step, ':', '\n', 'BEST=', self.best, '\n', 'min of fitness=', self.fmin)
        return self.best, self.fmin
