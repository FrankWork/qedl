# from __future__ import unicode_literals

import sys
from six.moves import xrange
import codecs
import os

def load_score(data_path):
  fin1 = open(data_path + "/new_test_data")
  fin2 = open(data_path + "/new_score")

  map = {}

  for line1 in fin1.readlines():
    vec = line1.strip().split("\t")
    segs = vec[0].split(" ||| ")
    st = int(vec[1])
    en = int(vec[2])
    label = int(vec[3])
    try:
      str_score = fin2.readline().strip()
      score = float(str_score)
    except:
      print(str_score)
      exit()
    sent = "".join(segs[1:-1])
    word = "".join(segs[st+1:en])
    if map.get(sent) == None:
      map[sent] = []
    map[sent].append((word, score, label))
  
  fin1.close()
  fin2.close()

  for key, value in map.items():
    map[key] = sorted(value, key=lambda elem:elem[1], reverse=True)
  return map

def remove_substr(words):
  keys = words.keys()
  remove = set()
  for string in keys:
    for substr in keys:
      if substr in string and substr != string:# substr is substring of string
        if words[substr] < words[string]:
          remove.add(substr)
        else:
          remove.add(string)
  for word in remove:
    words.pop(word)

def get_link_result(data_path, data, threshold):
  # fout = open(data_path + "/new_link_results", 'w')
  results = []

  top_n = 2

  for key, sorted_vec in data.items():
    top_n_entity = {}
    for i in xrange(len(sorted_vec)):
      word, score, label = sorted_vec[i]
      # if i < top_n:
			#   top_n_entity[sorted_vec[i][0]] = sorted_vec[i][1] 
      if score > threshold:
        top_n_entity[word] = score 

      # top_n_entity[word] = score 
    
    remove_substr(top_n_entity)

    idx = 0
    while(len(top_n_entity) == 0 and idx < len(sorted_vec)):
      word, score, label = sorted_vec[idx]
      top_n_entity[word] = score
      remove_substr(top_n_entity)
      idx += 1
    
    # idx = 0
    # while(len(top_n_entity) == 0 and idx < len(sorted_vec)):
    #   word, score, label = sorted_vec[idx]
    #   top_n_entity[word] = score
    #   idx += 1
    
    top_n_list = []
    for word, score in top_n_entity.items():
      top_n_list.append(word)
    
    if len(top_n_list) == 0:
      print(key)
      for idx in range(len(sorted_vec)):
        word, score, label = sorted_vec[idx]
        print(word)
    
    buf = key + "\t" + "|||".join(list(top_n_list))
    # fout.write(buf + '\n')
    results.append(buf)
  return results

def load_origin_test_sentence():
  '''
  return: a list of list
          e.g. lines[i] = [i-th sentence, 'NIL']
  '''
  orig_file = codecs.open("ccks1_test_dataset.csv", 'r', 'utf8')
  lines = []
  for line in orig_file:
    line = line.strip()
    idx = line.find(',')
    lines.append([line[idx+1:], 'NIL'])
  return lines

def write_sort_result(results, orig_sents, outfile):
  for idx in range(len(orig_sents)):
    orig_sents[idx][1] = 'nil'

  for sentence in results:  
    # sentence = sentence.strip()
    sentence = unicode(sentence.decode('utf8'))
    arr = sentence.split('\t')
    if len(arr) == 2:
      for id, pairs in enumerate(orig_sents):
        nospace = "".join([x for x in pairs[0] if x != ' '])
        if arr[0] == nospace.lower():
          if len(arr[1]) != 0:
            orig_sents[id][1] = arr[1]

  fout = codecs.open(outfile, 'w', 'utf8')
  for id, pairs in enumerate(orig_sents):
    fout.write(str(id) + ',' + pairs[0] + '\t' + pairs[1] + '\n')
  fout.close()
  # fout = codecs.open(outfile, 'w', 'utf8')
  # for id, pairs in enumerate(orig_sents):
  #   fout.write(str(id) + ',' + pairs[1] + '\t' + pairs[1] + '\n')
  # fout.close()

if __name__ == '__main__':
  data_path = sys.argv[1]
  data = load_score(data_path)

  orig_sents = load_origin_test_sentence()

  fin = codecs.open("f1_score", "r", "utf8")
  line = fin.readline()
  fin.close()
  threshold = float(line.split(',	')[0])
  
  # candidate_threshold = [x*0.001 for x in range(170, 300, 1)]
  # # for threshold in candidate_threshold:
  # threshold = 0.2 #0.1880 #0.2520 # 0.202, 0.272
  # threshold = 0
  results = get_link_result(data_path, data, threshold)
  write_sort_result(results, orig_sents, data_path + "/result.csv")

