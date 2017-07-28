import sys

label = sys.argv[2]

fin = open(sys.argv[1])

for line in fin.readlines():
	query, entity = line.strip().split("\t")
	seg_vec = []
	seg_vec.append("<begin>")
	seg_vec.extend(query.split(" ||| "))
	seg_vec.append("<end>")
	for st in xrange(0, len(seg_vec)):
		for en in xrange(st + 1, len(seg_vec) + 1):
			word = "".join(seg_vec[st:en])
			if word == entity:
				print " ||| ".join(seg_vec) + "\t" + str(st - 1) + "\t" + str(en) + "\t" + label
	
