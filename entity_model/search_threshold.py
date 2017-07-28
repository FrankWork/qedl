# from __future__ import unicode_literals

import sys
from six.moves import xrange
import codecs
import os


def load_data(data_path):
  fin1 = open(data_path + "/test_data")
  fin2 = open(data_path + "/score")

  map = {}

  for line1 in fin1.readlines():
    vec = line1.strip().split("\t")
    segs = vec[0].split(" ||| ")
    st = int(vec[1])
    en = int(vec[2])
    label = int(vec[3])
    score = float(fin2.readline().strip())
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
  

def f1_score(data, threshold):
  top_n = 2

  n_tp = 0.0 # true positive
  n_pred = 0.0 # tp + fp
  n_real = 0.0 # tp + fn
  
  for key, sorted_vec in data.items():
    top_n_entity = {}
    labeled_entity = set()
    for i in xrange(len(sorted_vec)):
      word, score, label = sorted_vec[i]
      if score > threshold:
        top_n_entity[word] = score 
      if label == 1:
        labeled_entity.add(word)
    
    remove_substr(top_n_entity)

    idx = 0
    while(len(top_n_entity) == 0 and idx < len(sorted_vec)):
      word, score, label = sorted_vec[idx]
      top_n_entity[word] = score
      remove_substr(top_n_entity)
      idx += 1

    n_pred += len(top_n_entity)
    n_real += len(labeled_entity)

    for e in top_n_entity:
      if e in labeled_entity:
        n_tp += 1

  try:
    p = n_tp / n_pred
    r = n_tp / n_real
    f = 2*p*r / (p+r)
  except ZeroDivisionError:
    p,r,f = .0, .0, .0
  
  return threshold, p, r, f


if __name__ == '__main__':
  data_path = sys.argv[1]
  data = load_data(data_path)
  candidate_threshold = [x*0.001 for x in range(1, 1000, 1)]
  # candidate_threshold = [x*0.001 for x in range(170, 300, 1)]
  # candidate_threshold = [x*0.0001 for x in range(1, 10000, 1)]
  
  best = 0
  best_res = None
  for thres in candidate_threshold:
    res = f1_score(data, thres)
    print("%.4f,\t%.4f,\t%.4f,\t%.4f" % res)
    f1 = res[-1]
    if best < f1:
      best = f1
      best_res = res
  print("score,\tp,\tr,\tf1")
  print("%.4f,\t%.4f,\t%.4f,\t%.4f" % best_res)

  fout = codecs.open("f1_score", "w", "utf8")
  fout.write("%.4f,\t%.4f,\t%.4f,\t%.4f\n" % best_res)
  fout.close()
  
