import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
 
TRAIN_URL="http://download.tensorflow.org/data/iris_training.csv"
train_path=tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

COLUMN_NAMES=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
df_iris=pd.read_csv(train_path,names=COLUMN_NAMES,header=0)
iris=np.array(df_iris)

fig=plt.figure('Iris Data',figsize=(15,15))
fig.suptitle("Anderson's Iris Data Set\n(Bule->Setosa|Red->Versicolor|Green->Virginica)",fontsize=20)

for i in range(4):
    for j in range(4):
        plt.subplot(4,4,4*i+(j+1))
        if(i==j):
            plt.hist(iris[:,j], align='mid', color='blue', edgecolor='black')
            #plt.bar(iris[i,j],iris[i,i],width=0.8,facecolor='blue',edgecolor='black')
        else:
            plt.scatter(iris[:,j],iris[:,i],c=iris[:,4],cmap='brg')
        if(i==0):
             plt.title(COLUMN_NAMES[j])
        if(j==0):
            plt.ylabel(COLUMN_NAMES[i])

plt.show()

