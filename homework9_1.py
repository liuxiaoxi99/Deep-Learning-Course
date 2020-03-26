import tensorflow as tf
import numpy as np

x1=tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21],dtype="float64")
x2=tf.constant([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2],dtype="float64")
y=tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30],dtype="float64")
x0=tf.constant(np.ones(len(x1)),dtype="float64")

X=tf.stack((x0,x1,x2),axis=1)
Y=tf.reshape(y,(-1,1))

Xt=tf.transpose(X)
XtX_1=tf.linalg.inv(tf.matmul(Xt,X))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
XtX_1_Xt=tf.matmul(XtX_1,Xt)
W=tf.matmul(XtX_1_Xt,Y)

W=tf.reshape(W,(-1,1))

print("——————————开始预估房价——————————")
x1_test=int(input("请输入商品房的面积（20~500）："))
if x1_test<20 :
    x1_test=int(input("输入数据超出范围，请重新输入商品房的面积（20~500）:"))
elif(x1_test>500):
   x1_test=int(input("输入数据超出范围，请重新输入商品房的面积（20~500）:"))

x2_test=int(input("请输入商品房的房间数（1~10）："))
if x2_test<1:
    x2_test=int(input("输入数据超出范围，请重新输入商品房的房间数（1~10）:"))
elif x2_test>10:
    x2_test=int(input("输入数据超出范围，请重新输入商品房的房间数（1~10）:"))
y_guess=W[1]*x1_test+W[2]*x2_test+W[0]

print("根据你输入的信息，预测到该商品房的房价是：",np.array(y_guess),"万元")
