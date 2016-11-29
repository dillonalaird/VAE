from __future__ import division
from __future__ import print_function

from datetime import datetime

import os
import math
import tensorflow as tf


class VAE(object):
    def __init__(self, config, sess, model_dir):
        self.x_dim      = config.x_dim
        self.h_dim      = config.h_dim
        self.z_dim      = config.z_dim
        self.lr         = config.lr
        self.epochs     = config.epochs
        self.batch_size = config.batch_size
        self.sess       = sess
        self.model_dir  = model_dir

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
        self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01,
                                                        dtype=tf.float32)
        self.build_variables()
        self.build_model()

    def build_variables(self):
        with tf.variable_scope("encoder"):
            self.W3 = tf.get_variable("W3", shape=[self.x_dim, self.h_dim],
                                      initializer=self.initializer)
            self.W4 = tf.get_variable("W4", shape=[self.h_dim, self.z_dim],
                                      initializer=self.initializer)
            self.W5 = tf.get_variable("W5", shape=[self.h_dim, self.z_dim],
                                      initializer=self.initializer)
            self.b3 = tf.get_variable("b3", shape=[self.h_dim],
                                      initializer=self.initializer)
            self.b4 = tf.get_variable("b4", shape=[self.z_dim],
                                      initializer=self.initializer)
            self.b5 = tf.get_variable("b5", shape=[self.z_dim],
                                      initializer=self.initializer)

        with tf.variable_scope("decoder"):
            self.W1 = tf.get_variable("W1", shape=[self.z_dim, self.h_dim],
                                      initializer=self.initializer)
            self.W2 = tf.get_variable("W2", shape=[self.h_dim, self.x_dim],
                                      initializer=self.initializer)
            self.b1 = tf.get_variable("b1", shape=[self.h_dim],
                                      initializer=self.initializer)
            self.b2 = tf.get_variable("b2", shape=[self.x_dim],
                                      initializer=self.initializer)

    def encode(self, x):
        with tf.variable_scope("encoder"):
            h        = tf.tanh(tf.matmul(x, self.W3) + self.b3)
            mu       = tf.matmul(h, self.W4) + self.b4
            mu2      = tf.square(mu)
            log_sig2 = tf.matmul(h, self.W5) + self.b5
            sig2     = tf.exp(log_sig2)
            sig      = tf.exp(0.5*log_sig2)
            eps      = tf.random_normal([self.batch_size, self.z_dim], mean=0.0, stddev=1.0)
            z = mu + sig*eps
        return log_sig2, sig2, mu2, z

    def decode(self, z):
        with tf.variable_scope("decoder"):
            y = tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(z, self.W1) + self.b1),
                                     self.W2) + self.b2)
        return y

    def build_model(self):
        log_sig2, sig2, mu2, z = self.encode(self.x)
        y = self.decode(z)
        log_pz = tf.reduce_sum(self.x*tf.log(y) + (1 - self.x)*tf.log(1 - y), 1)
        KL = -0.5*tf.reduce_sum(1 + log_sig2 - mu2 - sig2, 1)

        self.sampled = self.decode(self.z)
        self.loss = -tf.reduce_sum(KL + log_pz)/self.batch_size
        self.optim = tf.contrib.layers.optimize_loss(self.loss, None, self.lr,
                "Adagrad", summaries=["learning_rate", "loss", "gradient_norm"])
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def sample(self, z):
        return self.sess.run([self.sampled], feed_dict={self.z: z})[0]

    def train(self, data):
        num_train   = data.train.num_examples
        num_valid   = data.validation.num_examples
        train_iters = int(math.floor(num_train/self.batch_size))
        valid_iters = int(math.floor(num_valid/self.batch_size))
        best_valid  = float("inf")
        merged_sum  = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/{}".format(self.model_dir),
                                        self.sess.graph)

        for epoch in xrange(self.epochs):
            train_loss = 0.
            for itr in xrange(train_iters):
                x, _ = data.train.next_batch(self.batch_size)
                outputs = self.sess.run([self.optim, self.loss, merged_sum],
                                        feed_dict={self.x: x})
                train_loss += outputs[1]
                if itr % 2 == 0:
                    writer.add_summary(outputs[-1], train_iters*epoch + itr)
            print("[Train] [Time: {}] [Neg. Log Likelihood: {}]"
                  .format(datetime.now(), train_loss/train_iters))
            valid_loss = 0.
            for _ in xrange(valid_iters):
                x, _ = data.validation.next_batch(self.batch_size)
                loss, = self.sess.run([self.loss], feed_dict={self.x: x})
                valid_loss += loss
            print("[Valid] [Time: {}] [Neg. Log Likelihood: {}]"
                  .format(datetime.now(), valid_loss/valid_iters))
            if loss < best_valid:
                best_valid = loss
                self.saver.save(self.sess, os.path.join("checkpoints",
                                                        self.model_dir,
                                                        "bestvalid"))

    def test(self, data):
        num_test   = data.test.num_examples
        test_iters = int(math.floor(num_test/self.batch_size))
        test_loss = 0.
        for _ in xrange(test_iters):
            x, _  = data.test.next_batch(self.batch_size)
            loss, = self.sess.run([self.loss], feed_dict={self.x: x})
            test_loss += loss

        print("[Test] [Neg. Log Likelihood: {}]".format(test_loss/test_iters))

    def load(self):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(os.path.join("checkpoints",
                                                          self.model_dir))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("[!] No checkpoint found")
