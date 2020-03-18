import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt  
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()


plt.figure(figsize=(8,8))
for i in range(16):
    num=np.random.randint(1,10000)
    plt.subplot(4,4,i+1)
    plt.rcParams['font.sans-serif']="SimHei"
    plt.axis("off")
    plt.imshow(test_x[num],cmap="gray")
    plt.title("标签值："+str(test_y[num]),fontsize=14)
    plt.suptitle("MNIST测试集样本",color="red",fontsize=20)

plt.show()

