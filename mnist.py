import tensorflow as tf
import numpy as np
import os

# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers


class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = layers.Conv2D(filters=20, kernel_size=3, strides=1, padding='same', activation='relu')
        self.drop1 = layers.Dropout(rate=0.5)
        self.pool1 = layers.MaxPool2D(2, 2)
        self.conv2 = layers.Conv2D(filters=20, kernel_size=3, strides=1, padding='same', activation='relu')
        self.drop2 = layers.Dropout(rate=0.5)
        self.pool2 = layers.MaxPool2D(2, 2)
        self.conv3 = layers.Conv2D(filters=10, kernel_size=3, strides=1, padding='same', activation='relu')
        self.pool3 = layers.AveragePooling2D(7, 7)

    def call(self, x):
        x = self.conv1(x)
        x = self.drop1(x, training=True)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drop2(x, training=True)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = tf.squeeze(x)
        return x


def loss_fn(m, x, y):
    y_pred = m(x)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_pred)
    return loss


def acc_fn(m, x, y):
    preds = m(x).numpy()
    acc = np.sum(np.argmax(preds, axis=1) == y.numpy(), dtype=np.float32) / x.numpy().shape[0]
    return acc


def prepare_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_data = np.expand_dims((X_train.astype(np.float32) / 255.).astype(np.float32), axis=-1)
    test_data = np.expand_dims((X_test.astype(np.float32) / 255.).astype(np.float32), axis=-1)
    train_labels, test_labels = y_train.astype(np.int), y_test.astype(np.int)
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(60000)
    dataset = dataset.batch(128, drop_remainder=True)

    return dataset, test_data, test_labels


def train(model, opt, dataset, test_data, test_labels):
    for epoch in range(1):
        losses = []
        for (batch, (inp, targ)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = loss_fn(model, inp, targ)
            gradients = tape.gradient(loss, model.trainable_variables)
            # print("loss: ", loss.numpy(), ",\tacc: ", acc_fn(model, inp, targ)*100, "%")
            opt.apply_gradients(zip(gradients, model.variables))
            losses.append(loss.numpy())

        print("Epoch :", epoch, ", train loss :", np.mean(losses))
        acc = acc_fn(model, test_data, test_labels)
        print("Epoch :", epoch, ", valid acc:", acc * 100, "%")
    print("train done.\n")
    model.save_weights('weights.h5')
    print("\n\n")


if __name__ == '__main__':
    # load the data
    dataset, test_data, test_labels = prepare_data()
    test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)

    # prepare the model
    lenet = LeNet()
    y = lenet(tf.convert_to_tensor(np.random.random((10, 28, 28, 1)), tf.float32))
    learning_rate = tf.Variable(1e-3, name="learning_rate")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train(lenet, optimizer, dataset, test_data, test_labels)

