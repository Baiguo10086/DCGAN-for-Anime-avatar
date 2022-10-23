import glob
import io
import math
import time

import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential, Input, Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
# from scipy.misc import imread, imsave
from imageio import imread, imsave
from scipy.stats import entropy

K.set_image_data_format('channels_last')

np.random.seed(1337)


def generator():
    gen = Sequential()

    gen.add(Dense(2048, activation='relu', input_dim=100))
    # gen.add(Dense(input_dim=100, output_dim=2048))
    # gen.add(LeakyReLU(alpha=0.2))

    gen.add(Dense(256 * 8 * 8))
    # 批归一化
    gen.add(BatchNormalization())
    gen.add(Activation('relu'))
    # 重构 height, width, chns
    gen.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    # 上采样
    gen.add(UpSampling2D(size=(2, 2)))
    # 卷积 并保持大小不变
    gen.add(Conv2D(128, (5, 5), padding='same'))
    gen.add(Activation('relu'))

    gen.add(UpSampling2D(size=(2, 2)))

    gen.add(Conv2D(64, (5, 5), padding='same'))
    gen.add(Activation('relu'))

    gen.add(UpSampling2D(size=(2, 2)))

    gen.add(Conv2D(3, (5, 5), padding='same'))
    gen.add(Activation('tanh'))
    # gen.summary()
    return gen


def discriminator():
    dis = Sequential()
    dis.add(
        Conv2D(128,
               (5, 5),
               padding='same',
               input_shape=(64, 64, 3),
               activation=LeakyReLU(alpha=0.2))
    )
    # dis.add(LeakyReLU(alpha=0.2))
    dis.add(MaxPooling2D(pool_size=(2, 2)))

    dis.add(Conv2D(256, (3, 3)))
    dis.add(LeakyReLU(alpha=0.2))
    dis.add(MaxPooling2D(pool_size=(2, 2)))

    dis.add(Conv2D(512, (3, 3)))
    dis.add(LeakyReLU(alpha=0.2))
    dis.add(MaxPooling2D(pool_size=(2, 2)))

    dis.add(Flatten())
    dis.add(Dense(1024))
    dis.add(LeakyReLU(alpha=0.2))

    dis.add(Dense(1))
    dis.add(Activation('sigmoid'))
    # dis.summary()
    return dis


# 对抗模型
def adversarial_model(gen, dis):
    model = Sequential()
    model.add(gen)
    # 冻结 discriminator 权重不再更新
    dis.trainable = False
    model.add(dis)
    return model


# # use_board(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)
# def use_board(callback, name, loss, batch_no):
#     """
#     可视化内容
#     """
#     # for name, value in zip(names, logs):
#     # summary = tf.Summary()
#     tf.compat.v1.disable_eager_execution()
#     summary = tf.compat.v1.Summary()
#     summary_value = summary.value.add()
#     summary_value.simple_value = loss
#     summary_value.tag = name
#     file_writer = tf.summary.create_file_writer(callback.log_dir)
#     # file_writer.add_summary(summary, batch_no)
#     file_writer.flush()


def save_rgb_img(img, path):
    # Save a rgb image
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("RGB Image")

    plt.savefig(path)
    plt.close()


def train():
    start_time = time.time()
    dataset_dir = "image60000/*.*"
    batch_size = 128
    z_shape = 100
    epochs = 400
    dis_learning_rate = 0.0002  # 0.005
    gen_learning_rate = 0.0003
    dis_momentum = 0.4  # 0.5
    gen_momentum = 0.5
    dis_nesterov = True
    gen_nesterov = True

    # Load images
    all_images = []
    for index, filename in enumerate(glob.glob(dataset_dir)):  # 获取图像名字作为list
        all_images.append(imread(filename))  # 读取图像

    # 标准化所有图像  可以再改进一下
    x = np.array(all_images)

    # 归一化
    x = (x - 127.5) / 127.5
    x = x.astype(np.float32)

    # 随机梯度下降算法 -- 优化器
    gen_optimizer = SGD(learning_rate=gen_learning_rate, momentum=gen_momentum, nesterov=gen_nesterov)
    dis_optimizer = SGD(learning_rate=dis_learning_rate, momentum=dis_momentum, nesterov=dis_nesterov)

    gen_model = generator()
    # 均方误差 作为loss func
    gen_model.compile(optimizer=gen_optimizer, loss='binary_crossentropy')

    dis_model = discriminator()
    # 二元交叉熵 作为loss func
    dis_model.compile(optimizer=dis_optimizer, loss='binary_crossentropy')

    adversarial = adversarial_model(gen_model, dis_model)
    adversarial.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    # 可视化显示损失
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_graph=True, write_images=True
                              # write_grads=True,
                              )
    tensorboard.set_model(gen_model)
    tensorboard.set_model(dis_model)

    for epoch in range(epochs):
        print("--------------------------")
        print("Epoch:{}".format(epoch))

        dis_losses = []
        gen_losses = []

        num_batches = int(x.shape[0] / batch_size)

        print("Number of batches:{}".format(num_batches))
        for index in range(num_batches):
            print("Batch:{}".format(index))

            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

            generated_images = gen_model.predict_on_batch(z_noise)

            # 训练辨别器 dis

            dis_model.trainable = True

            image_batch = x[index * batch_size:(index + 1) * batch_size]

            y_real = np.ones((batch_size,)) * 0.9
            y_fake = np.zeros((batch_size,)) * 0.1

            # y_real = np.ones(batch_size, ) - np.random.random_sample(batch_size) * 0.2
            # y_fake = np.random.random_sample(batch_size) * 0.2

            dis_loss_real = dis_model.train_on_batch(image_batch, y_real)
            dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)

            d_loss = (dis_loss_real + dis_loss_fake) / 2
            print("d_loss:", d_loss)

            dis_model.trainable = False

            # 对抗model 中训练gen
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

            g_loss = adversarial.train_on_batch(z_noise, y_real)
            print("g_loss:", g_loss)

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

        # 10代保存一张image
        if epoch % 10 == 0:
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            gen_images1 = gen_model.predict_on_batch(z_noise)

            for img in gen_images1[:2]:
                save_rgb_img(img, "results/three_{}.png".format(epoch))

        print("Epoch:{}, dis_loss:{}".format(epoch, np.mean(dis_losses)))
        print("Epoch:{}, gen_loss: {}".format(epoch, np.mean(gen_losses)))

        # 保存 loss 可视化 每一代 参数变化
        # use_board(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        # use_board(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

    # 保存模型
    gen_model.save("generator_model3.h5")
    dis_model.save("generator_model3.h5")

    print("Time:", (time.time() - start_time))


if __name__ == '__main__':
    train()
