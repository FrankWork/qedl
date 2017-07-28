# -*- coding:utf-8 -*- - 
import sys
# from six.moves import xrange
# import jieba
reload(sys)
sys.setdefaultencoding('utf-8')

data_path = 'word_seg_'+sys.argv[2]+ '/'

fn = open(data_path + sys.argv[1] + "_neg_data", "w")

fin = open("mentions")
mset = set()
for line in fin.readlines():
	mset.add(line.strip("\n").lower())

fin = open(data_path + 'output_test.txt')

for line in fin:
	seg_vec = line.strip().lower().split()
	wset = set()

	for st in xrange(len(seg_vec)):
		for en in xrange(st + 1, len(seg_vec) + 1):
			word = str("".join(seg_vec[st:en]))
			if (word in mset):
				wset.add(word)
	# if len(wset) == 0:
	# 	print line.stirp() + "\t" + " ".join(seg_vec)
	for w in wset:
		fn.write(" ||| ".join(seg_vec) + "\t" + w + '\n')
