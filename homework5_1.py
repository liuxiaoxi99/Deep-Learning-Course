import numpy as np
np.random.seed(612)
a=np.random.rand(1000,)
m=int(input("请输入一个1~100之间的数字："))
n=0
print("序号   索引值   随机数")
for i in range(0,1000,1):
    if (i%m==0):
        print(n,end="\t")
        print(i,end="\t")
        print(a[i])
        n+=1
    else:
        continue