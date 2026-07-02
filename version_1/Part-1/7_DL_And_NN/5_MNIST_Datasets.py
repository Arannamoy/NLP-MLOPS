import tensorflow as tf
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(x_test.shape)


plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.axis("off")
    plt.title(str(y_train[i]))

plt.tight_layout()
plt.savefig("./DL_And_NN/Images/six.png")

