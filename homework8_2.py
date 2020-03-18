import tensorflow as tf 
x=tf.constant([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=tf.constant([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])

sum1=0
for i in range(10):
    sum1+=(x[i]*y[i])
sum1*=10             #等式1中的分子的第一部分的值
sum2=sum_x=sum_y=0
for i in range(10):
    sum_x+=x[i]
    sum_y+=y[i]
sum2=sum_x*sum_y     #等式1中的分子的第二部分的值
sum3=0
for i in range(10):
    sum3+=(x[i]**2)
sum3*=10             #等式1中的分母的第一部分的值

w=(sum1-sum2)/(sum3-sum_x**2)
b=(sum_y-w*sum_x)/10

print("输出w的值为：",end="")
print(w)
print("输出b的值为：",end="")
print(b)