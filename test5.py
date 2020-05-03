import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#导入数据集
mnist=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#对图像images归一化处理
train_images=train_images/255.0
test_images=test_images/255.0

#对标签labels进行独热编码
train_labels_ohe=tf.one_hot(train_labels,depth=10).numpy()
test_labels_ohe=tf.one_hot(test_labels,depth=10).numpy()

#建立Sequential线性堆叠模型
model=tf.keras.models.Sequential()

#添加平坦层
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

#添加隐藏层
model.add(tf.keras.layers.Dense(units=64,kernel_initializer='normal',activation='relu'))
model.add(tf.keras.layers.Dense(units=32,kernel_initializer='normal',activation='relu'))

#添加输出层
model.add(tf.keras.layers.Dense(10,activation='softmax'))

#模型输出摘要
model.summary()

#定义训练模式
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#设置训练参数
train_epochs=30
batch_size=35

#训练模型
train_history=model.fit(train_images,train_labels_ohe,validation_split=0.2,epochs=train_epochs,batch_size=batch_size,verbose=2)

#训练过程指标可视化
def show_train_history(train_history,train_metric,val_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[val_metric])
    plt.title('Train History')
    plt.ylabel(train_metric)
    plt.xlabel('Epoch')
    plt.show()
show_train_history(train_history,'loss','val_loss')
show_train_history(train_history,"accuracy","val_accuracy")