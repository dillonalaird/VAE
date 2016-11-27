from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from variational_autoencoder import VAE
from scipy.stats import norm

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


flags = tf.app.flags

flags.DEFINE_integer("h_dim", 512, "The hidden size [512]")
flags.DEFINE_integer("z_dim", 2, "The size of the latent space [2]")
flags.DEFINE_integer("epochs", 12, "The number of epochs [12]")
flags.DEFINE_integer("batch_size", 128, "The batch size [128]")
flags.DEFINE_float("lr", 0.01, "The learning rate [0.01]")
flags.DEFINE_boolean("test", False, "Run algorithm on test set [False]")
flags.DEFINE_boolean("sample", False, "Sample random data point [False]")
flags.DEFINE_boolean("sample_manifold", False, "Sample the unit square on z [False]")

FLAGS = flags.FLAGS


def sample_image(vae):
    z = [np.random.multivariate_normal(np.zeros((vae.z_dim,)), np.eye(vae.z_dim))]
    image = vae.sample(z)
    image = image.reshape((28, 28))
    image *= 255.
    plt.imshow(image.astype(np.uint8), cmap="gray")
    plt.show()


def sample_manifold2d(vae, N):
    image = np.zeros((N*28, N*28))
    for z1 in xrange(N):
        for z2 in xrange(N):
            z = [np.array([norm.ppf(z1*(1/N) + 1/(2*N)),
                           norm.ppf(z2*(1/N) + 1/(2*N))])]
            sample = vae.sample(z).reshape((28, 28))
            image[z1*28:(z1 + 1)*28,z2*28:(z2 + 1)*28] = sample

    image *= 255.
    plt.imshow(image.astype(np.uint8), cmap="gray")
    plt.show()


def get_model_dir(config, exceptions):
    attrs = config.__dict__["__flags"]
    keys = list(attrs.keys())
    keys.sort()

    names = ["{}={}".format(key, attrs[key]) for key in keys if key not in exceptions]
    model_dir = os.path.join(*names)
    ckpt_dir = os.path.join("checkpoints", model_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return model_dir


def main(_):
    config = FLAGS
    # TODO: hardcoded for mnist
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    config.x_dim = 784

    with tf.Session() as sess:
        vae = VAE(config, sess, get_model_dir(config, ["test", "sample", "sample_manifold", "batch_size"]))
        if config.sample:
            vae.load()
            sample_image(vae)
        elif config.sample_manifold:
            vae.load()
            sample_manifold2d(vae, 20)
        elif config.test:
            vae.load()
            vae.test(mnist)
        else:
            vae.train(mnist)


if __name__ == "__main__":
    tf.app.run()
