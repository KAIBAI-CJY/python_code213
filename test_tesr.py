import numpy as np
import matplotlib.pyplot as plt

EE = 2000  # 弹性模量

cutmax = 0.12  # 最大应力或应变
cutmin = 0  # 最小应力或应变
i = 0  # 增量步计数
ncyele = 3  # 循环周次
e = 0  # 应变初始化
ep = 0  # 塑性应变初始化
sy = 100  # 居脸应力
s = 0  # 应力初始化
gama = 0  # 塑性流动率初始化
de0 = 5e-6  # 位移增量

E = []
S = []

for cycle in range(1, ncyele + 1):
    print(cycle)  # 打印循环
    while e < cutmax:
        i += 1
        de = de0
        e += de
        s_trial = EE * (e - ep)
        f = np.abs(s_trial) - sy
        if f < 0:
            s = s_trial
            gama = 0
        else:
            s = sy * np.sign(s)
            gama = de
        dep = gama
        ep += dep
        S.append(s)
        E.append(e)
    while e > cutmin:
        i += 1
        de = -de0
        e += de
        s_trial = EE * (e - ep)
        f = np.abs(s_trial) - sy
        if f < 0:
            s = s_trial
            gama = 0
        else:
            s = sy * np.sign(s)
            gama = de
        dep = gama
        ep += dep
        S.append(s)
        E.append(e)

E = np.array(E)
S = np.array(S)

plt.plot(E, S, 'm')
plt.show()
