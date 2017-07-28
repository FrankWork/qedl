#!/home/zhanghan/python-2.7.9/bin/python

import model
import config
import tensorflow as tf
import reader
import time
import sys
import copy

tf.flags.DEFINE_string("data_path", "", "Data path")
tf.flags.DEFINE_integer("batch_size", 20, "")
tf.flags.DEFINE_integer("lstm_layer_num", 1, "")
tf.flags.DEFINE_integer("lstm_hidden_size", 128, "")
tf.flags.DEFINE_integer("epoch_num", 103, "")
tf.flags.DEFINE_float("keep_prob", 0.7, "")
FLAGS = tf.flags.FLAGS

def predict(session, rs_model, para_conf, data_reader):
	start_time = time.time()
	fetches = {"loss": rs_model._loss, "prob": rs_model._total_prob}

	batches_per_epoch = data_reader._data_set.batches_per_epoch
	
	batches = data_reader._data_set.data_num // para_conf.batch_size
	last_batch_size = data_reader._data_set.data_num % para_conf.batch_size 
	if last_batch_size != 0:
		batches += 1
	#else:
	#	last_batch_size = para_conf.batch_size
		
	for step in xrange(batches):
		batch_size = para_conf.batch_size
		#if step == batches - 1:
		#	batch_size = last_batch_size
		batch_inputs = data_reader.get_batch(batch_size)
		feed_dict = {}
		feed_dict[rs_model._sent_word_inputs] = batch_inputs.sent_word_inputs
		# feed_dict[rs_model._sent_len] = batch_inputs.sent_len
		feed_dict[rs_model._sent_char_inputs] = batch_inputs.sent_char_inputs
		# feed_dict[rs_model._word_len] = batch_inputs.word_len
		feed_dict[rs_model._st_poses] = batch_inputs.st_poses
		feed_dict[rs_model._en_poses] = batch_inputs.en_poses
		feed_dict[rs_model._labels] = batch_inputs.labels
		feed_dict[rs_model._batch_size] = len(batch_inputs.sent_word_inputs)

		ret = session.run(fetches, feed_dict=feed_dict)
		score_arr = ret["prob"]
		for score in score_arr:
			print score		

def main(_):
	para_conf = config.ParameterConfig()
	sys_conf = config.SysConfig()
	sys_conf.data_path = FLAGS.data_path

	para_conf.batch_size = FLAGS.batch_size
	para_conf.lstm_layer_num = FLAGS.lstm_layer_num
	para_conf.lstm_hidden_size = FLAGS.lstm_hidden_size
	para_conf.epoch_num = FLAGS.epoch_num
	para_conf.keep_prob = FLAGS.keep_prob
	
	word_embed_file = sys_conf.data_path + "/" + sys_conf.word_embedding_data
	char_embed_file = sys_conf.data_path + "/" + sys_conf.char_dict_data
	train_file = sys_conf.data_path + '/' + sys_conf.train_data
	valid_file = sys_conf.data_path + '/' + sys_conf.valid_data
	test_file = sys_conf.data_path + '/' + sys_conf.test_data

	global_dict = reader.load_global_embedding(word_embed_file, char_embed_file)

	train_set = reader.load_data_set(para_conf.batch_size, global_dict, train_file)
	valid_set = reader.load_data_set(para_conf.batch_size, global_dict, valid_file)
	test_set = reader.load_data_set(para_conf.batch_size, global_dict, test_file)

	train_reader, valid_reader, test_reader = reader.get_data_readers(para_conf,
																global_dict, train_set, valid_set, test_set)
	
	eval_para_conf = copy.copy(para_conf)#config.ParameterConfig()
	eval_para_conf.keep_prob = 1.0
	eval_para_conf.epoch_num = 1
	eval_para_conf.batch_size = 20
	
	with tf.Graph().as_default():
		with tf.variable_scope("model", reuse=None):
			rs_model = model.EntityScoreModel()
			rs_model.generate_embeddings(global_dict, para_conf)
			rs_model.inference(para_conf, is_train=True)
			rs_model.loss(para_conf)
			rs_model.train(para_conf)

		with tf.variable_scope("model", reuse=True):
			eval_model = model.EntityScoreModel()
			eval_model.generate_embeddings(global_dict, eval_para_conf)
			eval_model.inference(eval_para_conf, is_train=False)
			eval_model.loss(eval_para_conf)
	
		saver = tf.train.Saver()
		sv = tf.train.Supervisor()
		with sv.managed_session() as session:
			saver.restore(session, FLAGS.data_path + "/model/model.iter100")
			session.run([rs_model._word_embeddings])
			#session.run([rs_model._word_embeddings, rs_model._char_embeddings])
			predict(session, eval_model, eval_para_conf, test_reader)


if __name__ == "__main__":
	tf.app.run()
