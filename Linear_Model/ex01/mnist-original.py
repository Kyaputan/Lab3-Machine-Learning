from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_raw = loadmat("Lab3 Machine Learning\ex01\mnist-original.mat")

mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
x=mnist["data"]
y=mnist["target"]

number = x[10000]
number_image = number.reshape(28,28)
plt.imshow(number_image,cmap=plt.cm.binary,interpolation="nearest")
plt.show()