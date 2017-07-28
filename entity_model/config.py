class ParameterConfig(object):
	def __init__(self):
		self.word_embedding_size = 256
		self.char_embedding_size = 25
		self.max_sent_len = 20
		self.max_word_len = 7
		self.batch_size = 20 
		self.lstm_layer_num = 1 
		self.lstm_hidden_size = 128 
		self.lstm_maxpool_win = 4 
		self.conv_win_size2featrue_size={2:200, 3:200}
		self.learning_rate = 0.1
		self.keep_prob = 0.7 
		self.epoch_num = 110 

class SysConfig(object):
	data_path = "data/"
	train_data = "train_data"
	valid_data = "valid_data"
	test_data =  "test_data"
	word_embedding_data = "word_embedding"
	char_dict_data = "char_embedding_25"

class GlobalDict(object):
	word_num = 0
	char_num = 0
	word2id = {}
	id2word = {}
	char2id = {}
	id2char = {}
	word_embeddings = []
	char_embeddings = []

class DataSet(object):
	def __init__(self):
		self.sent_word_data = []
		self.sent_char_data = []
		self.sent_len = []
		self.word_len = []
		self.st_poses = []
		self.en_poses = []
		self.labels = []
		data_num = 0
		batches_per_epoch = 0
