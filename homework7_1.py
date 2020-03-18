import matplotlib.pyplot as plt 
from PIL import Image

img=Image.open('D:\\DeepLearning\\lena.tiff')
img_r,img_g,img_b=img.split()
plt.figure(figsize=(10,10))
plt.suptitle("图像基本操作",color="blue",fontsize=20)
plt.rcParams['font.sans-serif']="SimHei"

plt.subplot(221)   #子图1
plt.axis("off")
img_r1=img_r.resize((50,50))
plt.imshow(img_r1,cmap="gray")
plt.title("R-缩放",fontsize=14)

plt.subplot(222)   #子图2
#plt.axis("off")
img_g1=img_g.transpose(Image.FLIP_LEFT_RIGHT)
img_g2=img_g1.transpose(Image.ROTATE_270)
plt.imshow(img_g2,cmap="gray")
plt.title("G-镜像+旋转",fontsize=14)

plt.subplot(223)   #子图3
plt.axis("off")
img_b1=img_b.crop((0,0,150,150))
plt.imshow(img_b1,cmap="gray")
plt.title("B-裁剪",fontsize=14)

img_rgb=Image.merge("RGB",[img_r,img_g,img_b])
plt.subplot(224)   #子图4
plt.axis("off")
plt.imshow(img_rgb)
plt.title("RGB",fontsize=14)
plt.show()

img_rgb.save("test.png")












