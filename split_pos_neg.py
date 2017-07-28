# -*- coding:utf-8 -*- - 
import sys
# from six.moves import xrange
# import jieba
reload(sys)
sys.setdefaultencoding('utf-8')

data_path = 'word_seg_'+sys.argv[2]+ '/'
f_seg_input = open('input.txt')
f_seg_output = open(data_path + 'output.txt')

seg_dict = {}
for line in f_seg_input:
	seg = f_seg_output.readline().strip().lower().split()
	seg_dict[line.strip().lower()] = seg

f_seg_input.close()
f_seg_output.close()

fp = open(data_path + sys.argv[1] + "_pos_data", "w")
fn = open(data_path + sys.argv[1] + "_neg_data", "w")
# fe = open("error", "w")
fe = open(data_path + "error", "a")

fin = open("mentions")
mset = set()
for line in fin.readlines():
	mset.add(line.strip("\n").lower())

def adjust_seg_res(seg_vec, labels):
	ret = []
	for word in seg_vec:
		tag = False
		word = str(word)
		for l in labels:
			pos = word.find(l)
			if pos != -1 and len(word) != len(l):
				pre = word[:pos]
				if pre != "":
					pre = pre.decode("utf8")
					for c in pre:
						ret.append(c.encode("utf8"))
				ret.append(l)

				suf = word[pos+len(l):]
				if suf != "":
					suf = suf.decode("utf8")
					for c in suf:
						ret.append(c.encode("utf8"))
				tag = True
				break
		if not tag:
			ret.append(word)

	return ret

total = 0
fin = open(sys.argv[1] + "_org")

for line in fin:
	vec = line.strip().lower().split("\t")
	# seg_vec = list(jieba.cut(vec[0]))
	seg_vec = seg_dict[vec[0]]

	try:
		labels = vec[1].split("|||")
	except:
		# FIXME ignored some data
		# print('except here')
		# print line
		# exit()
		continue

	for i in xrange(len(labels)):
		labels[i] = labels[i].strip()
	
	seg_vec = adjust_seg_res(seg_vec, labels)

	lset = set(labels)
	wset = set()
	# neg_set = set()

	for st in xrange(len(seg_vec)):
		for en in xrange(st + 1, len(seg_vec) + 1):
			word = str("".join(seg_vec[st:en]))
			if (word in mset) or (word in lset):
				wset.add(word)
			# else:
			# 	neg_set.add(word)

	pset = set()

	total += len(labels)

	for l in labels:
		if l in wset:
			pset.add(l)
		else:
			fe.write(" ".join(seg_vec) + "\t" + l + '\n')

	for w in wset:
		if w in pset:
			fp.write(" ||| ".join(seg_vec) + "\t" + w + '\n')
		else:
			fn.write(" ||| ".join(seg_vec) + "\t" + w + '\n')
	
	# for w in neg_set:
	# 	fn.write(" ||| ".join(seg_vec) + "\t" + w + '\n')