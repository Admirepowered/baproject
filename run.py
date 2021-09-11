import numpy as np
from ba import BA


# 测试用例,Griewan函数，x=(0, 0...,0)处有全局极小值 随机生成三维地图
def func(x):
    y1 = 1 / 4000 * sum(np.power(x, 2))
    y2 = 1
    for h in range(x.size):#10个维度
        y2 = y2 * np.cos(x[h] / np.sqrt(h + 1))
        #print(y2,end='Bw123\n')
    y = y1 - y2 + 1
    #print(y,end='DDDD\n')
    #print(y,end='TT')
    #print(x)
    return y


if __name__ == '__main__':
    #ba = BA(10, 20, 100, 0, 2, -2, 2, func)
    ba = BA(10, 20, 1, 0, 2, -2, 2, func)#初始化蝙蝠算法参数 10维度 20个个体 100次迭代 Qmini Qmax B属于[-2,2]
    best, fmin = ba.start_iter()
    print('=============================================')
    print('BEST=', best, '\n', 'min of fitness=', fmin)

    #(x-a)^2+(y-b)^2=R^2

    #fi=fmin+(fmax-fmin)xβ (1)
    #vi^t=vi^(t-1)+(xi^t-x*)xfi (2)
    #xi^t=xi^(t-1)+vi^(t) (3)
    #智能优化算法：黄金正弦算法

    #xi=
