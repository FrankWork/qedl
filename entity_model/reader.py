#! -*- encoding: utf-8 -*-

import tensorflow as tf
import config
import random

def read_word_embedding(embed_file, global_dict):
	fin = open(embed_file)
	global_dict.word2id = {"<null>" : 0, "<unk>" : 1, "<begin>" : 2, "<end>" : 3, " " : 4}
	global_dict.id2word = {0 : "<null>", 1 : "<unk>", 2 : "<begin>", 3 : "<end>", 4 : " "}
	cur_id = 5
	for line in fin:
		arr = line.strip().split(" ")
		global_dict.word2id[arr[0]] = cur_id
		global_dict.id2word[cur_id] = arr[0]
		global_dict.word_embeddings.append([])
		for i in xrange(1, len(arr)):
			 global_dict.word_embeddings[cur_id - 5].append(float(arr[i]))
		cur_id += 1
	fin.close()

	global_dict.word_num = cur_id
	# print "loaded " + str(global_dict.word_num) + " words"


def read_char_embedding(embed_file, global_dict):
	fin = open(embed_file)
	global_dict.char2id = {"<null>" : 0, "<unk>" : 1, "<begin>" : 2, "<end>" : 3, " " : 4}
	global_dict.id2char = {0 : "<null>", 1 : "<unk>", 2 : "<begin>", 3 : "<end>", 4 : " "}
	cur_id = 5
	for line in fin:
		arr = line.strip().split(" ")
		if arr[0] == '<unk>':
			continue
		global_dict.char2id[arr[0]] = cur_id
		global_dict.id2char[cur_id] = arr[0]
		global_dict.char_embeddings.append([])
		for i in xrange(1, len(arr)):
			 global_dict.char_embeddings[cur_id - 5].append(float(arr[i]))
		cur_id += 1
	fin.close()

	global_dict.char_num = cur_id
	# print "loaded " + str(global_dict.char_num) + " chars"

def load_global_embedding(word_embed_file, char_embed_file):
	"""load word_embedding, char_embedding, word2id, id2word, char2id, id2char
	"""
	global_dict = config.GlobalDict()
	read_word_embedding(word_embed_file, global_dict)
	read_char_embedding(char_embed_file, global_dict)
	return global_dict


def load_word_data(words, word2id, sent_len, word_data):
	unk_id = 1
	words_id = []
	n = len(words)
	sent_len.append(n)
	for w in words:
		words_id.append(word2id.get(w, unk_id))
	word_data.append(words_id)

def load_char_data(words, char2id, words_len, char_data):
	# python2
	# '我'.decode('utf8') == u'我'
	# u'我'.encode('utf8') == '我'
	# len('我爱北京天安门')21
	# len(u'我爱北京天安门')7
	unk_id = 1
	ids = []
	lengths = []
	for word in words:
		if word in ["<null>", "<unk>", "<begin>", "<end>", " "]:
			ids.append([char2id.get(word, unk_id)])
			lengths.append(1)
			continue

		uword = word.decode('utf8')
		lengths.append(len(uword))
		ids_in_word = []
		for uchar in uword:
			char = uchar.encode('utf8')
			ids_in_word.append(char2id.get(char, unk_id))
		ids.append(ids_in_word)

	char_data.append(ids)
	words_len.append(lengths)


def load_data_set(batch_size, global_dict, path):
	data_set = config.DataSet()
	fin = open(path)
	for line in fin.readlines():
		arr = line.strip().split("\t")
		words = arr[0].split(" ||| ")
		load_word_data(words, global_dict.word2id, data_set.sent_len, data_set.sent_word_data)
		load_char_data(words, global_dict.char2id, data_set.word_len, data_set.sent_char_data)

		data_set.st_poses.append(int(arr[1]))
		data_set.en_poses.append(int(arr[2]))
		data_set.labels.append(int(arr[3]))
	fin.close()

	data_set.data_num = len(data_set.sent_word_data)
	data_set.batches_per_epoch = data_set.data_num // batch_size
	if data_set.data_num % batch_size != 0:
		data_set.batches_per_epoch += 1
	return data_set


def get_max_len(para_conf, train_set, valid_set, test_set):
	"""get `para_conf.max_sent_len`, `para_conf.max_word_len` from data_set
	"""
	max_n = 0
	for sent_len in [train_set.sent_len, valid_set.sent_len, test_set.sent_len]:
		for n in sent_len:
			if max_n < n:
				max_n = n
	para_conf.max_sent_len = max_n

	max_n = 0
	for word_len in [train_set.word_len, valid_set.word_len, test_set.word_len]:
		for lengths in word_len:
			for n in lengths:
				if max_n < n:
					max_n = n
	para_conf.max_word_len = max_n
	# len_counter = {}
	# if word_len in len_counter:
	# 	len_counter[word_len] += 1
	# else:
	# 	len_counter[word_len] = 1
	# print(len_counter)
	# {1: 25273, 2: 32642, 3: 5689, 4: 2166, 5: 9061, 6: 167, 7: 8845, 8: 35, 9: 9, 10: 4, 11: 3, 12: 8, 13: 6}

def padding_data_set(para_conf, train_set, valid_set, test_set):
	padding_id = 0

	max_len = para_conf.max_sent_len
	for data_set in [train_set, valid_set, test_set]:
		for n, sent in zip(data_set.sent_len, data_set.sent_word_data):
			assert n == len(sent)
			for _ in xrange(n, max_len):
				sent.append(padding_id)
	
	max_word_len = para_conf.max_word_len
	for data_set in [train_set, valid_set, test_set]:
		for lengths, words in zip(data_set.word_len, data_set.sent_char_data):
			assert len(lengths) == len(words)
			for _ in range(len(words), max_len):
				lengths.append(0)
				words.append([])
			for n, word in zip(lengths, words):
				assert n == len(word)
				for _ in range(n, max_word_len):
					word.append(padding_id)
				assert max_word_len == len(word)
				

def get_data_readers(para_conf, global_dict, train_set, valid_set, test_set):
	get_max_len(para_conf, train_set, valid_set, test_set)
	padding_data_set(para_conf, train_set, valid_set, test_set)

	# for data_set in [train_set, valid_set, test_set]:
	# 	for n, sent in zip(data_set.sent_len, data_set.sent_word_data):
	# 		if n < 0:
	# 			print('<')
	# 			exit()
	# print('>')
	# exit()

	train_reader = Reader(train_set, is_train=True)
	valid_reader = Reader(valid_set, is_train=False)
	test_reader = Reader(test_set, is_train=False)

	return train_reader, valid_reader, test_reader

class BatchInputs(object):
	def __init__(self):
		self.sent_word_inputs = []
		self.sent_len = []
		self.sent_char_inputs = []
		self.word_len = []
		self.st_poses = []
		self.en_poses = []
		self.labels = []


class Reader(object):
	def __init__(self, data_set, is_train):
		self._data_set = data_set
		self._is_train = is_train
		self._shuffle_pool = range(data_set.data_num)
		if (is_train):
			random.shuffle(self._shuffle_pool)
		self._ptr = 0
	
	def get_batch(self, batch_size):
		batch_inputs = BatchInputs()
		for i in xrange(batch_size):
			index = self._shuffle_pool[self._ptr]
			batch_inputs.sent_word_inputs.append(self._data_set.sent_word_data[index])
			batch_inputs.sent_len.append(self._data_set.sent_len[index])
			batch_inputs.sent_char_inputs.append(self._data_set.sent_char_data[index])
			batch_inputs.word_len.append(self._data_set.word_len[index])
			batch_inputs.st_poses.append((self._data_set.st_poses[index]))
			batch_inputs.en_poses.append((self._data_set.en_poses[index]))
			batch_inputs.labels.append((self._data_set.labels[index]))
			
			self._ptr += 1
			if self._ptr >= self._data_set.data_num:
				self._ptr = 0
				if (self._is_train):
					random.shuffle(self._shuffle_pool)
				break
			
		return batch_inputs


