import tensorflow as tf
import config
import reader
from tensorflow.python.training import moving_averages

def get_variable_on_cpu(name, shape, dtype, initializer):
	with tf.device('/cpu:0'):
		return tf.get_variable(name, shape, dtype=dtype, initializer=initializer)

def get_weights_variable(name, shape, stddev):
	return get_variable_on_cpu(name, shape, tf.float32,
	    tf.truncated_normal_initializer(stddev=stddev))

def get_biases_variable(name, shape, value):
	return get_variable_on_cpu(name, shape, tf.float32,
	    tf.constant_initializer(value, tf.float32))

def lstm_cell(name, para_conf, is_train):
	cell = tf.contrib.rnn.BasicLSTMCell(para_conf.lstm_hidden_size,
	    forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
	if is_train and para_conf.keep_prob < 1:
		cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=para_conf.keep_prob)
	return cell
		

class EntityScoreModel(object):

	def generate_embeddings(self, global_dict, para_conf):
		self._word_num = global_dict.word_num
		self._char_num = global_dict.char_num
		with tf.variable_scope("embeddings"):
			null_embedding = tf.get_variable("null", shape=[1, para_conf.word_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
			unk_embedding = tf.get_variable("unk", shape=[1, para_conf.word_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
			begin_embedding = tf.get_variable("begin", shape=[1, para_conf.word_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
			end_embedding = tf.get_variable("end", shape=[1, para_conf.word_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
			space_embedding = tf.get_variable("space", shape=[1, para_conf.word_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))

			self._word_embbeddings = [null_embedding, unk_embedding, begin_embedding, end_embedding, space_embedding]
			self._word_embbeddings.append(tf.convert_to_tensor(global_dict.word_embeddings, dtype=tf.float32))
			self._word_embeddings = tf.concat(self._word_embbeddings, axis=0, name="word_embeddings")
			

			null_char_embedding = tf.get_variable("null_char", shape=[1, para_conf.char_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
			unk_char_embedding = tf.get_variable("unk_char", shape=[1, para_conf.char_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
			begin_char_embedding = tf.get_variable("begin_char", shape=[1, para_conf.char_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
			end_char_embedding = tf.get_variable("end_char", shape=[1, para_conf.char_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
			space_char_embedding = tf.get_variable("space_char", shape=[1, para_conf.char_embedding_size],
			    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))

			self._char_embeddings = [null_char_embedding, unk_char_embedding, begin_char_embedding, end_char_embedding, space_char_embedding]
			self._char_embeddings.append(tf.convert_to_tensor(global_dict.char_embeddings, dtype=tf.float32))
			self._char_embeddings = tf.concat(self._char_embeddings, axis=0, name="char_embeddings")

			# self._char_embeddings = tf.get_variable("char_embeddings", dtype=tf.float32,
			#    shape=[global_dict.char_num, para_conf.char_embedding_size],
			#    initializer=tf.random_uniform_initializer(-0.1, 0.1))
		
	def inference(self, para_conf, is_train):
		# if is_train:
		# 	self.mode = "train"
		# else:
		# 	self.mode = "predict"
		# self._extra_train_ops = []

		word_embeddings = self._word_embeddings
		char_embeddings = self._char_embeddings

		self._sent_word_inputs = tf.placeholder(tf.int32, [None, para_conf.max_sent_len])
		self._sent_len = tf.placeholder(tf.int32, [None])
		self._sent_char_inputs = tf.placeholder(tf.int32, [None, para_conf.max_sent_len, 
														         para_conf.max_word_len])
		self._word_len = tf.placeholder(tf.int32, [None, para_conf.max_sent_len])

		# [batch_size, max_len, word_emb_size]
		sent_inputs = tf.nn.embedding_lookup(word_embeddings, self._sent_word_inputs)
		# [batch_size, max_len, word_len, char_emb_size]
		sent_char_inputs = tf.nn.embedding_lookup(char_embeddings, self._sent_char_inputs)

		def rnn_cell(name):
			def cell():
				_cell = tf.contrib.rnn.BasicLSTMCell(25)
				if is_train and para_conf.keep_prob < 1:
					_cell = tf.contrib.rnn.DropoutWrapper(_cell, output_keep_prob=para_conf.keep_prob)
				return _cell
			with tf.variable_scope(name+'rnn_cell'):
				_rnn_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in xrange(para_conf.lstm_layer_num)])
			return _rnn_cell

		char_fw_cell = rnn_cell("char_lstm_fw_cell")
		char_bw_cell = rnn_cell("char_lstm_bw_cell")
			    
		char_outputs = [] # [max_len, batch_size, out_size]
		for i in range(para_conf.max_sent_len):
			input = sent_char_inputs[:, i, :, :] #[batch_size, max_word_len, char_emb_size]
			# length = self._word_len[:, i]
			output, _ = tf.nn.bidirectional_dynamic_rnn(char_fw_cell, char_bw_cell, input, dtype=tf.float32)
			char_outputs.append(tf.concat([output[0][:, -1, :], output[1][:, 0, :]], axis=-1))
		# [batch_size, max_len, out_size]
		char_outputs = tf.transpose(tf.convert_to_tensor(char_outputs), perm=[1, 0, 2])
		# char_outputs = self._batch_norm('char_outputs', char_outputs)

		sent_inputs = tf.concat([sent_inputs, char_outputs], axis=-1)

		# input_size = para_conf.word_embedding_size
		#input_size = para_conf.max_word_len * para_conf.char_embedding_size + para_conf.word_embedding_size 		

		with tf.variable_scope("sent_lstm_layer"):
			sent_lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(
			    [lstm_cell("sent_lstm_fw_cell", para_conf, is_train) for _ in xrange(para_conf.lstm_layer_num)],
			    )
			sent_lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(
			    [lstm_cell("sent_lstm_bw_cell", para_conf, is_train) for _ in xrange(para_conf.lstm_layer_num)],
			    )
			sent_lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(sent_lstm_fw_cell, sent_lstm_bw_cell,
			    sent_inputs, dtype=tf.float32)
			sent_lstm_outputs = tf.concat(sent_lstm_outputs, axis=2)
		# sent_lstm_outputs = self._batch_norm('sent_lstm_outputs', sent_lstm_outputs)

		softmax_input_size = 2 * para_conf.lstm_hidden_size
		softmax_inputs = tf.reshape(sent_lstm_outputs, [-1, softmax_input_size])
		with tf.variable_scope("softmax_layer"):
			softmax_weights = get_weights_variable("weights", [softmax_input_size, 3], 0.1)
			softmax_biases = get_biases_variable("biases", [3], 0.0)
		self._softmax_outputs = tf.nn.softmax(tf.matmul(softmax_inputs, softmax_weights) + softmax_biases)
		self._softmax_outputs = tf.reshape(self._softmax_outputs, [-1, para_conf.max_sent_len, 3])


	def loss(self, para_conf):
		self._st_poses = tf.placeholder(tf.int32, [None])
		self._en_poses = tf.placeholder(tf.int32, [None])
		self._labels = tf.placeholder(tf.int32, [None])
		self._batch_size = tf.placeholder(tf.int32, [])
		
		indices = tf.range(self._batch_size)
		st_poses = tf.stack([indices, self._st_poses], axis=1)
		en_poses = tf.stack([indices, self._en_poses], axis=1)
		st_poses = tf.sparse_to_dense(st_poses, [self._batch_size, para_conf.max_sent_len], 1.0, 0.0)
		en_poses = tf.sparse_to_dense(en_poses, [self._batch_size, para_conf.max_sent_len], 1.0, 0.0)
		self._st_prob = st_poses * self._softmax_outputs[:, :, 0]
		self._en_prob = en_poses * self._softmax_outputs[:, :, 1]
		self._st_prob = tf.reduce_sum(self._st_prob, axis=1)
		self._en_prob = tf.reduce_sum(self._en_prob, axis=1)	

		self._total_prob = self._st_prob * self._en_prob
		self._loss = -(tf.reduce_mean(tf.cast(self._labels, tf.float32) * 
		    tf.log(tf.clip_by_value(self._total_prob, 1e-10, 1.0)))
		    + tf.reduce_mean(tf.cast(1 - self._labels, tf.float32) *
		    tf.log(tf.clip_by_value(1 - self._total_prob, 1e-10, 1.0))))

	def train(self, para_conf):
		self._lr = tf.placeholder(tf.float32, [])
		sgd_op = tf.train.GradientDescentOptimizer(self._lr).minimize(self._loss)
		self._train_op = sgd_op
		# train_ops = [sgd_op] + self._extra_train_ops
		# self._train_op = tf.group(*train_ops)

	# TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
	def _batch_norm(self, name, x):
		"""Batch normalization."""
		with tf.variable_scope(name):
			params_shape = [x.get_shape()[-1]]

			beta = tf.get_variable(
				'beta', params_shape, tf.float32,
				initializer=tf.constant_initializer(0.0, tf.float32))
			gamma = tf.get_variable(
				'gamma', params_shape, tf.float32,
				initializer=tf.constant_initializer(1.0, tf.float32))

			if self.mode == 'train':
				mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

				moving_mean = tf.get_variable(
					'moving_mean', params_shape, tf.float32,
					initializer=tf.constant_initializer(0.0, tf.float32),
					trainable=False)
				moving_variance = tf.get_variable(
					'moving_variance', params_shape, tf.float32,
					initializer=tf.constant_initializer(1.0, tf.float32),
					trainable=False)

				self._extra_train_ops.append(moving_averages.assign_moving_average(
					moving_mean, mean, 0.9))
				self._extra_train_ops.append(moving_averages.assign_moving_average(
					moving_variance, variance, 0.9))
			else:
				mean = tf.get_variable(
					'moving_mean', params_shape, tf.float32,
					initializer=tf.constant_initializer(0.0, tf.float32),
					trainable=False)
				variance = tf.get_variable(
					'moving_variance', params_shape, tf.float32,
					initializer=tf.constant_initializer(1.0, tf.float32),
					trainable=False)
				# tf.summary.histogram(mean.op.name, mean)
				# tf.summary.histogram(variance.op.name, variance)
			# elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
			y = tf.nn.batch_normalization(
				x, mean, variance, beta, gamma, 0.001)
			y.set_shape(x.get_shape())
			return y