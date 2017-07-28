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

def calculate_f1(ac_map, top_n):
	total = 0.0
	correct = 0.0
	pos_label_num = 0.0
	
	for _, vec in ac_map.items():
		sorted_vec = sorted(vec, key=lambda elem:elem[0], reverse=True)
		total += min(top_n, len(vec))
		for i in xrange(len(vec)):
			if i < top_n:
				correct += sorted_vec[i][1]
			pos_label_num += sorted_vec[i][1]
	#print "correct:" + str(correct)

	acu = correct / total
	rec = correct / pos_label_num
	f1 = 2 * acu * rec / (acu + rec)
	return (acu, rec, f1)

def run_epoch(session, rs_model, para_conf, data_reader, is_train, lr=0.05):
	costs = 0.0
	examples = 0
	accuracy = 0.0
	start_time = time.time()
	fetches = {"loss": rs_model._loss, "prob": rs_model._total_prob}
	if (is_train):
		fetches["train_op"] = rs_model._train_op

	batches_per_epoch = data_reader._data_set.batches_per_epoch

	ac_map = {}

	top_n = 2

	for step in xrange(batches_per_epoch):
		batch_inputs = data_reader.get_batch(para_conf.batch_size)
		feed_dict = {}
		feed_dict[rs_model._sent_word_inputs] = batch_inputs.sent_word_inputs
		# feed_dict[rs_model._sent_len] = batch_inputs.sent_len
		feed_dict[rs_model._sent_char_inputs] = batch_inputs.sent_char_inputs
		# feed_dict[rs_model._word_len] = batch_inputs.word_len
		feed_dict[rs_model._st_poses] = batch_inputs.st_poses
		feed_dict[rs_model._en_poses] = batch_inputs.en_poses
		feed_dict[rs_model._labels] = batch_inputs.labels
		feed_dict[rs_model._batch_size] = len(batch_inputs.sent_word_inputs)
		if (is_train):
			feed_dict[rs_model._lr] = lr

		ret = session.run(fetches, feed_dict=feed_dict)
		costs += ret["loss"] * len(batch_inputs.sent_word_inputs)
		examples += len(batch_inputs.sent_word_inputs)
		if is_train and step % (batches_per_epoch // 10) == 10:
			print("%.3f cost: %.3f speed: %.0f qps" %
			    (step * 1.0 / batches_per_epoch, costs / examples,
			    examples / (time.time() - start_time)))
			sys.stdout.flush()

		probs = ret["prob"]
		for index in xrange(len(batch_inputs.sent_word_inputs)):
			if ac_map.get(tuple(batch_inputs.sent_word_inputs[index])) == None:
				ac_map[tuple(batch_inputs.sent_word_inputs[index])] = []
			ac_map[tuple(batch_inputs.sent_word_inputs[index])].append((probs[index], batch_inputs.labels[index]))

	eval_res = calculate_f1(ac_map, top_n)
	print "Top" + str(top_n) + " Accuracy of this epoch:" + str(eval_res[0])
	print "Top" + str(top_n) + " Recall of this epoch:" + str(eval_res[1])
	print "Top" + str(top_n) + " F1 of this epoch:" + str(eval_res[2])
	print "Cost of this epoch:" + str(costs / examples)
	sys.stdout.flush() 

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

	valid_set.batches_per_epoch = valid_set.data_num // eval_para_conf.batch_size
	if valid_set.data_num % eval_para_conf.batch_size != 0:
		valid_set.batches_per_epoch += 1

	test_set.batches_per_epoch = test_set.data_num // eval_para_conf.batch_size
	if test_set.data_num % eval_para_conf.batch_size != 0:
		test_set.batches_per_epoch += 1

	with tf.Graph().as_default():
		with tf.variable_scope("model"):
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
			# saver.restore(session,  FLAGS.data_path + "/model/model.iter50")
			session.run([rs_model._word_embeddings])
			lr = para_conf.learning_rate
			for step in xrange(para_conf.epoch_num):
				if (step + 1) >= 20 and (step + 1) % 3 == 0:
					lr *= 0.6
				if lr < 0.0005:
					lr = 0.0005
				print "Start epoch " + str(step + 1) + " learning rate: " + str(lr) 
				run_epoch(session, rs_model, para_conf, train_reader, is_train=True, lr=lr)

				if (step + 1) % 10 == 0:
					print "Valid_Set:"
					run_epoch(session, eval_model, eval_para_conf,
					    valid_reader, is_train=False)

					print "Test_Set:"
					run_epoch(session, eval_model, eval_para_conf,
					    test_reader, is_train=False)

				if (step + 1) % 5 == 0:
					saver.save(session, FLAGS.data_path + "/model/model.iter" + str(step + 1))
				sys.stdout.flush() 

if __name__ == "__main__":
	tf.app.run()
