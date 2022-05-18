import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, learning_rate=1e-3, batch_size=10000,
        reg=None, model_path=None, restore_path=None):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.iter = 0
        weights = self._initialize_weights()
        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 3])
        x_input = self.x
        print('input', x_input.shape)
        latent, shape = self.encoder(x_input, weights)
        print('latent', latent.shape)
        self.z = tf.reshape(latent, [batch_size, -1])
        self.x_r = self.decoder(latent, weights, shape)
        print('x_r', self.x_r.shape)
        self.saver = tf.train.Saver()
        # cost for reconstruction
        # l_2 loss 
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))   # choose crossentropy or l2 loss
        self.loss = self.cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)   # GradientDescentOptimizer # AdamOptimizer
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 3, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))
        iter_i = 1
        while iter_i < n_layers:
            enc_name_wi = 'enc_w' + str(iter_i)
            all_weights[enc_name_wi] = tf.get_variable(enc_name_wi, shape=[self.kernel_size[iter_i], self.kernel_size[iter_i],
                                                                           self.n_hidden[iter_i - 1], self.n_hidden[iter_i]],
                                                # initializer=tf.keras.initializers.glorot_normal, regularizer=self.reg)
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
            enc_name_bi = 'enc_b' + str(iter_i)
            all_weights[enc_name_bi] = tf.Variable(tf.zeros([self.n_hidden[iter_i]], dtype=tf.float32))  # , name = enc_name_bi
            iter_i = iter_i + 1
        iter_i = 1
        while iter_i < n_layers:
            dec_name_wi = 'dec_w' + str(iter_i - 1)
            all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers - iter_i], self.kernel_size[n_layers - iter_i],
                                                                           self.n_hidden[n_layers - iter_i - 1], self.n_hidden[n_layers - iter_i]],
                                                # initializer=tf.keras.initializers.glorot_normal, regularizer=self.reg)
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
            dec_name_bi = 'dec_b' + str(iter_i - 1)
            all_weights[dec_name_bi] = tf.Variable(tf.zeros([self.n_hidden[n_layers - iter_i - 1]], dtype=tf.float32))  # , name = dec_name_bi
            iter_i = iter_i + 1

        dec_name_wi = 'dec_w' + str(iter_i - 1)
        all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0], 3, self.n_hidden[0]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        dec_name_bi = 'dec_b' + str(iter_i - 1)
        all_weights[dec_name_bi] = tf.Variable(tf.zeros([1], dtype=tf.float32))  # , name = dec_name_bi
        return all_weights
    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1, 2, 2, 1], padding='SAME'), weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())
        n_layers = len(self.n_hidden)
        iter_i = 1
        while iter_i < n_layers:
            layeri = tf.nn.bias_add(
                tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[1, 2, 2, 1], padding='SAME'),
                weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())
            iter_i = iter_i + 1
        layer3 = layeri
        return layer3, shapes
    # Building the decoder
    def decoder(self, z, weights, shapes):
        n_layers = len(self.n_hidden)
        layer3 = z
        iter_i = 0
        while iter_i < n_layers:
            shape_de = shapes[n_layers - iter_i - 1]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack([tf.shape(self.x)[0], shape_de[1], shape_de[2], shape_de[3]]), \
            strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
            iter_i = iter_i + 1
        return layer3
    def partial_fit(self, X):
        cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        self.iter = self.iter + 1
        return cost
    def save_model(self, it):
        save_path = self.saver.save(self.sess, self.model_path, global_step=it)#
        print("model saved in file: %s" % save_path)
        # print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        # save_path = self.saver.save(self.sess, os.path.join(save_path, model_name), global_step=it)
    def restore(self):
        # self.saver.restore(self.sess, self.restore_path)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("model restored")
        else:
            print("from scratch")
def train_face(Img, CAE, n_input, batch_size):
    it = 0
    display_step = 500
    save_step = 500
    _index_in_epoch = 0
    _epochs = 0
    CAE.restore()
    # train the network
    while it < 2500:
        batch_x = Img
        batch_x = np.reshape(batch_x, [batch_size, n_input[0], n_input[1], 3])
        cost = CAE.partial_fit(batch_x)
        it = it + 1
        # print(it)
        avg_cost = cost/(batch_size)
        if it % display_step == 0:
            print("epoch: %.1d" % it)
            print("cost: %.8f" % avg_cost)
        if it % save_step == 0:
            CAE.save_model(it)
    return
if __name__ == '__main__':
    data = sio.loadmat('/srv/hd2/xuyikun/backup/2022AAAI/CIFAR10/CIFAR10_4000space.mat')
    Img = data['trainImages'].astype(np.float32)/255
    Label = data['trainLabels_space']
    Label = np.array(Label)
    n_input = [32, 32]
    kernel_size = [3, 3]
    n_hidden = [16]  # [4]   # [15], 32
    batch_size = Img.shape[0]
    lr = 1.0e-3  # learning rate
    model_path = '/srv/hd2/xuyikun/backup/2022AAAI/CIFAR10/results/preCIFAR10_4000/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path1 = model_path + 'CIFAR10.ckpt'
    model_name = 'CIFAR10'
    CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, learning_rate=lr, kernel_size=kernel_size,
                 batch_size=batch_size, model_path=model_path1, restore_path=model_path)
    train_face(Img, CAE, n_input, batch_size)