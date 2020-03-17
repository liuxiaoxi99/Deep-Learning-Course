import numpy as np 

x=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])

x_sum=0
y_sum=0
for i in range(10):
    x_sum+=x[i]
    x_avg=x_sum/10
for i in range(10):
    y_sum+=y[i]
    y_avg=y_sum/10

sum1=0
for i in range(10):
    sum1+=(x[i]-x_avg)*(y[i]-y_avg)
sum2=0
for i in range(10):
    sum2+=(x[i]-x_avg)**2

w=sum1/sum2
print("输出w的值为：",end="")
print(w)

print("输出b的值为：",end="")
print(y_avg-w*x_avg)



