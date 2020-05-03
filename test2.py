import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
x_data=np.linspace(0,100,500)
y_data=3.1234*x_data+2.98+np.random.randn(x_data.shape)*0.4
plt.scatter(x_data,y_data)
plt.plot(x_data,3.1234*x_data+2.98,"r")
def model(x,w,b):
    return tf.multiply(x,w)+b
w=tf.Variable(1.0,tf.float32)
b=tf.Variable(0.0,tf.float32)
def loss(x,y,w,b):
    err=model(x,w,b)-y
    squared_err=tf.square(err)
    return tf.reduce_mean(squared_err)
training_epochs=10
learning_rate=0.0001
def grad(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_=loss(x,y,w,b)
    return tape.gradient(loss_,[w,b])
step=0
loss_list=[]
display_step=20
for epoch in range(training_epochs):
    for xs,ys in zip(x_data,y_data):
        loss_=loss(xs,ys,w,b)
        loss_list.append(loss_)
        delta_w,delta_b=grad(xs,ys,w,b)
        change_w=delta_w*learning_rate
        change_b=delta_b*learning_rate
        w.assign_sub(change_w)
        b.assign_sub(change_b)
        step+=1
        if step % display_step == 0:
            print("Training Epoch:",'%d'%(epoch+1),"Step:%d"%(step),"loss=%f"%(loss_))
    plt.plot(x_data,w.numpy()*x_data+b.numpy())   
print("预测x=5.79时，y的值：",model(w,5.79,b))
plt.show()