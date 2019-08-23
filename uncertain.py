import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
from mnist import LeNet
import os

# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers


def uncertain(idx, model, data, target, T=500):
    rnd = np.random.randint(9999)
    sample = [np.argmax(model(data[rnd:rnd+1]).numpy(), axis=-1) for _ in range(T)]
    print("\n\nIndex = {}.".format(rnd))
    print(np.array(sample))
    #
    model_answer = collections.Counter(sample).most_common(3)
    print("Model answer = {}.".format(model_answer))
    print("Correct answer = {}.".format(target[rnd]))
    #
    height = [sample.count(j) / float(T) for j in range(10)]
    left = [j for j in range(10)]
    tick_label = [str(j) for j in range(10)]
    #
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    plt.bar(left=left, height=height, align='center', tick_label=tick_label)
    plt.tick_params(axis='y', which='major', labelsize=15)
    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.subplot(1, 2, 2)
    plt.imshow(data[rnd].numpy().reshape(28, 28))
    plt.title('{}\n'.format(target[rnd]), size=22)
    plt.axis('off')
    plt.savefig("mc/"+str(idx)+".jpg")
    plt.show()
    plt.close()


if __name__ == '__main__':
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    test_data = np.expand_dims((x_test.astype(np.float32) / 255.).astype(np.float32), axis=-1)
    test_labels = y_test.astype(np.int)

    lenet = LeNet()
    y = lenet(tf.convert_to_tensor(np.random.random((10, 28, 28, 1)), tf.float32))
    lenet.load_weights('weights.h5')
    print("load done.")
    for idx in range(100):
        uncertain(idx, lenet, tf.convert_to_tensor(test_data), test_labels)
