from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import math
from tensorflow.contrib import layers
from sklearn import cluster
from munkres import Munkres
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
class ConvAE(object):
	def __init__(self, n_input, kernel_size, n_hidden=None, reg_const1=1.0, reg_const2=1.0, reg=None, batch_size=256,
		denoise=False, model_path=None, logs_path='pretrain-model-fashion20/logs'):
		# n_hidden is a arrary contains the number of neurals on every layer
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.reg = reg
		self.model_path = model_path
		self.kernel_size = kernel_size
		self.iter = 0
		self.batch_size = batch_size
		weights = self._initialize_weights()
		# model
		self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 3])
		self.learning_rate = tf.placeholder(tf.float32, [])
		if denoise == False:
			x_input = self.x
			latent, shape = self.encoder(x_input, weights)

		else:
			x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x), mean=0, stddev=0.2, dtype=tf.float32))

			latent, shape = self.encoder(x_input, weights)
		self.z_conv = tf.reshape(latent, [batch_size, -1])
		self.z_ssc, Coef = self.selfexpressive_moduel(batch_size)
		self.Coef = Coef
		latent_de_ft = tf.reshape(self.z_ssc, tf.shape(latent))
		self.x_r_ft = self.decoder(latent_de_ft, weights, shape)

		self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
		self.cost_ssc = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.z_conv, self.z_ssc), 2))
		self.recon_ssc = tf.reduce_sum(tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
		self.reg_ssc = tf.reduce_sum(tf.pow(self.Coef, 2))
		tf.summary.scalar("ssc_loss", self.cost_ssc)
		tf.summary.scalar("reg_lose", self.reg_ssc)

		self.loss_ssc = self.cost_ssc*reg_const2 + reg_const1*self.reg_ssc + self.recon_ssc

		self.merged_summary_op = tf.summary.merge_all()
		self.optimizer_ssc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_ssc)
		self.init = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(self.init)
		self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	def _initialize_weights(self):
		all_weights = dict()
		n_layers = len(self.n_hidden)
		all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 3, self.n_hidden[0]],
			initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
		all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))
		iter_i = 1
		while iter_i < n_layers:
			enc_name_wi = 'enc_w' + str(iter_i)
			all_weights[enc_name_wi] = tf.get_variable(enc_name_wi, shape=[self.kernel_size[iter_i], self.kernel_size[iter_i], self.n_hidden[iter_i - 1], self.n_hidden[iter_i]],
												initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
			enc_name_bi = 'enc_b' + str(iter_i)
			all_weights[enc_name_bi] = tf.Variable(tf.zeros([self.n_hidden[iter_i]], dtype=tf.float32))  # , name = enc_name_bi
			iter_i = iter_i + 1
		iter_i = 1
		while iter_i < n_layers:
			dec_name_wi = 'dec_w' + str(iter_i - 1)
			all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers - iter_i], self.kernel_size[n_layers - iter_i], self.n_hidden[n_layers - iter_i - 1], self.n_hidden[n_layers - iter_i]],
													initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
			dec_name_bi = 'dec_b' + str(iter_i - 1)
			all_weights[dec_name_bi] = tf.Variable(tf.zeros([self.n_hidden[n_layers - iter_i - 1]], dtype=tf.float32))  # , name = dec_name_bi
			iter_i = iter_i + 1

		dec_name_wi = 'dec_w' + str(iter_i - 1)
		all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0], 3, self.n_hidden[0]],
												initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
		dec_name_bi = 'dec_b' + str(iter_i - 1)
		all_weights[dec_name_bi] = tf.Variable(tf.zeros([1], dtype=tf.float32))  # , name = dec_name_bi
		return all_weights
	# Building the encoder
	def encoder(self, x, weights):
		shapes = []
		shapes.append(x.get_shape().as_list())
		layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1, 2, 2, 1], padding='SAME'),
								weights['enc_b0'])
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
			layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack(
				[tf.shape(self.x)[0], shape_de[1], shape_de[2], shape_de[3]]), strides=[1, 2, 2, 1], padding='SAME'),
							weights['dec_b' + str(iter_i)])
			layer3 = tf.nn.relu(layer3)
			iter_i = iter_i + 1
		return layer3
	def selfexpressive_moduel(self, batch_size):
		Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')
		z_ssc = tf.matmul(Coef,	self.z_conv)
		return z_ssc, Coef
	def finetune_fit(self, X, lr):
		all_loss, C, l1_cost, l2_cost, summary, _ = self.sess.run((self.loss_ssc, self.Coef, self.reg_ssc, self.cost_ssc, self.merged_summary_op, self.optimizer_ssc), \
													feed_dict={self.x: X, self.learning_rate: lr})
		self.summary_writer.add_summary(summary, self.iter)
		self.iter = self.iter + 1
		return all_loss, C, l1_cost, l2_cost
	def initlization(self):
		tf.reset_default_graph()
		self.sess.run(self.init)
	def transform(self, X):
		return self.sess.run(self.z_conv, feed_dict={self.x: X})
	def save_model(self):
		save_path = self.saver.save(self.sess, self.model_path)
		print("model saved in file: %s" % save_path)
	def restore(self):
		self.saver.restore(self.sess, self.model_path)
		print("model restored")
def NMI(A,B):
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat
def best_map(L1,L2):
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1, nClass2)
	G = np.zeros((nClass, nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:, 1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2
def thrC(C,ro):
	if ro < 1:
		N = C.shape[1]
		Cp = np.zeros((N, N))
		S = np.abs(np.sort(-np.abs(C), axis=0))
		Ind = np.argsort(-np.abs(C), axis=0)
		for i in range(N):
			cL1 = np.sum(S[:, i]).astype(float)
			stop = False
			csum = 0
			t = 0
			while(stop == False):
				csum = csum + S[t, i]
				if csum > ro*cL1:
					stop = True
					Cp[Ind[0:t+1, i], i] = C[Ind[0:t+1, i], i]
				t = t + 1
	else:
		Cp = C
	return Cp
def post_proC(C, K, d, alpha):
	# C: coefficient matrix, K: number of clusters, d: dimension of each subspace
	n = C.shape[0]
	C = 0.5*(C + C.T)
	C = C - np.diag(np.diag(C)) + np.eye(n, n)  # for sparse C, this step will make the algorithm more numerically stable
	r = d*K + 1
	U, S, _ = svds(C, r, v0=np.ones(n))
	U = U[:, ::-1]
	S = np.sqrt(S[::-1])
	S = np.diag(S)
	U = U.dot(S)
	U = normalize(U, norm='l2', axis=1)
	Z = U.dot(U.T)
	Z = Z * (Z > 0)
	L = np.abs(Z ** alpha)
	L = L/L.max()
	L = 0.5 * (L + L.T)
	spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
	spectral.fit(L)
	grp = spectral.fit_predict(L) + 1
	return grp, L
def err_rate(gt_s, s):
	c_x = best_map(gt_s, s)
	# print(c_x)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	sio.savemat('/srv/hd2/xuyikun/backup/2022AAAI/CIFAR10/results/DSC_CIFAR10_4000space2.mat', mdict={'C': c_x})
	return missrate, NMI(gt_s, c_x)

data = sio.loadmat('/srv/hd2/xuyikun/backup/2022AAAI/CIFAR10/CIFAR10_4000space.mat')
I = data['trainImages'].astype(np.float32)/255
Label = data['trainLabels_space']
Img = np.array(I)
Img = np.reshape(Img, (Img.shape[0], 32, 32, 3))
n_input = [32, 32]
kernel_size = [3, 3]
n_hidden = [16]  # [15], 32
batch_size = Img.shape[0]
model_path0 = '/srv/hd2/xuyikun/backup/2022AAAI/CIFAR10/results/preCIFAR10_4000/'
if not os.path.exists(model_path0):
	os.makedirs(model_path0)
model_path = model_path0 + 'CIFAR10.ckpt-2500'
logs_path = model_path0 + 'logs'
num_class = 10  # how many class we sample
iter_ft = 0
ft_times = 5000
display_step = 5000
alpha = 0.045  # 0.09
learning_rate = 1e-3
reg1 = 1.0
reg2 = 0.7
CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_const1=reg1, reg_const2=reg2, kernel_size=kernel_size,
			 batch_size=batch_size, model_path=model_path, logs_path=logs_path)
acc_ = []
best={'acc':0, "cc1":-1, "cc2":-1, "epoch":-1, "alpha":-1}
for i in range(0, 1):
# for hypa1 in range(1,20,1):
# 	for hypa2 in range(1,10,1):
# 	alpha=i/100.
	coil20_all_subjs = Img
	coil20_all_subjs = coil20_all_subjs.astype(float)
	label_all_subjs = Label
	label_all_subjs = label_all_subjs - label_all_subjs.min() + 1
	label_all_subjs = np.squeeze(label_all_subjs)
	CAE.initlization()
	CAE.restore()
	for iter_ft in range(ft_times):
		iter_ft = iter_ft+1
		all_loss, C, l1_cost, l2_cost = CAE.finetune_fit(coil20_all_subjs, learning_rate)
		if iter_ft % display_step == 0:
			C = thrC(C, alpha)
			y_x, CKSym_x = post_proC(C, num_class, 4, 9)   # 5  11, 12 10 10
			missrate_x, NMI1 = err_rate(label_all_subjs, y_x)
			acc = 1 - missrate_x
			# print("experiment: %d" % i, "acc: %.4f" % acc)
			print("alpha:%.2f, epoch:%.1d" % (alpha, iter_ft), "cost: %.8f" % (all_loss / float(batch_size)),"acc: %.4f"%acc, "NMI: %.4f" % NMI1)
			if(acc>best['acc']):
				best={"acc": acc, "alpha": alpha, "epoch": iter_ft}
				# best={"acc":acc,"cc1":hypa1,"cc2":hypa2,"epoch":iter_ft}
	acc_.append(acc)
print("best:", best)
acc_ = np.array(acc_)
m = np.mean(acc_)
me = np.median(acc_)
print(m)
print(me)
print(acc_)