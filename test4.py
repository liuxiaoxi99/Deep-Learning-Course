import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

#获取数据集
mnist=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#数据集划分
total_num=len(train_images)        
valid_split=0.2     #验证集占比0.2
train_num=int(total_num*(1-valid_split))     #训练集数目

train_x=train_images[:train_num]     #训练集
train_y=train_labels[:train_num]

valid_x=train_images[train_num:]     #验证集
valid_y=train_labels[train_num:]

test_x=test_images
test_y=test_labels

#数据重塑形状
train_x=train_x.reshape(-1,784)
valid_x=valid_x.reshape(-1,784)
test_x=test_x.reshape(-1,784)

#数据特征归一化
train_x=tf.cast(train_x/255.0,tf.float32)
valid_x=tf.cast(valid_x/255.0,tf.float32)
test_x=tf.cast(test_x/255.0,tf.float32)

#标签数据进行独热编码
train_y=tf.one_hot(train_y,depth=10)
valid_y=tf.one_hot(valid_y,depth=10)
test_y=tf.one_hot(test_y,depth=10)

#构建模型
def model(x,w,b):
    pred=tf.matmul(x,w)+b
    return tf.nn.softmax(pred)

#定义模型变量
W=tf.Variable(tf.random.normal([784,10],mean=0.0,stddev=1.0,dtype=tf.float32))
B=tf.Variable(tf.zeros([10]),dtype=tf.float32)

#定义交叉熵损失函数
def loss(x,y,w,b):
    pred=model(x,w,b)
    loss_=tf.keras.losses.categorical_crossentropy(y_true=y,y_pred=pred)
    return tf.reduce_mean(loss_)

#定义训练超参数
training_epochs=40          #训练轮数
batch_size=30               #单次训练样本数
learning_rate=0.005         #学习率

#计算样本数据[x,y]在参数[w,b]点上的梯度
def grad(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_=loss(x,y,w,b)  
    return tape.gradient(loss_,[w,b])           #返回梯度向量

#定义优化器
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

#定义准确率
def accuracy(x,y,w,b):
    pred=model(x,w,b)
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#训练模型
total_step=int(train_num/batch_size)
loss_list_train=[]
loss_list_valid=[]
acc_list_train=[]
acc_list_valid=[]

for epoch in range(training_epochs):
    for step in range(total_step):
        xs=train_x[step*batch_size:(step+1)*batch_size]
        ys=train_y[step*batch_size:(step+1)*batch_size]

        grads=grad(xs,ys,W,B)
        optimizer.apply_gradients(zip(grads,[W,B]))
    loss_train=loss(train_x,train_y,W,B).numpy()
    loss_valid=loss(valid_x,valid_y,W,B).numpy()
    acc_trian=accuracy(train_x,train_y,W,B).numpy()
    acc_valid=accuracy(valid_x,valid_y,W,B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    acc_list_train.append(acc_trian)
    acc_list_valid.append(acc_valid)
    print("epoch={:3d} || train_loss={:.4f} || train_acc={:.4f} || val_loss={:.4f} || val_acc={:.4f}".format(
        epoch+1,loss_train,acc_trian,loss_valid,acc_valid))

#显示训练过程
plt.figure(figsize=(8,4))

plt.subplot(121)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train,'blue',label='Train Loss')
plt.plot(loss_list_valid,'red',label='Valid Loss')
plt.legend(loc=1)

plt.subplot(122)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(acc_list_train,'blue',label='Train Acc')
plt.plot(acc_list_valid,'red',label='Valid Acc')
plt.legend(loc=1)

plt.show()

#完成训练后，在测试集上评估模型的准确率
acc_test=accuracy(test_x,test_y,W,B).numpy()
print("Test accuracy(模型的准确率):",acc_test)

#定义预测函数
def predict(x,w,b):
    pred=model(x,w,b)
    result=tf.argmax(pred,1).numpy()
    return result

pred_test=predict(test_x,W,B)

#定义可视化函数
def plot_images_labels_prediction(images,labels,preds,index=0,num=10):
    fig=plt.gcf()      #获取当前图表，
    fig.set_size_inches(10,4) 
    if num>10:
        num=10
    for i in range(0,num):
        ax=plt.subplot(2,5,i+1)
        ax.imshow(np.reshape(images[index],(28,28)),cmap='binary')
        title="label="+str(labels[index])
        if len(preds)>0:
            title+=",predict="+str(preds[index])
        ax.set_title(title,fontsze=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index=index+1
    plt.show()

