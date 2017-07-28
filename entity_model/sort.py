import sys
import codecs

data_path = sys.argv[1]
orig_file = codecs.open("ccks1_test_dataset.csv", 'r', 'utf8')
res_file  = codecs.open(data_path + "/new_link_results", 'r', 'utf8')

lines = []
for line in orig_file:
  line = line.strip()
  idx = line.find(',')
  lines.append([line[idx+1:], 'NIL'])
  # id = line[:idx]

# for id, pairs in enumerate(lines):
#   print(str(id) + ',' + pairs[0] + '\t' + pairs[1])

for sentence in res_file:
  sentence = sentence.strip()
  arr = sentence.split('\t')
  if len(arr) == 1:
    # print(sentence)
    pass
  else:
    for id, pairs in enumerate(lines):
      nospace = "".join([x for x in pairs[0] if x != ' '])
      if arr[0] == nospace.lower():
        # print(type(arr[0]))
        # print(type(nospace))
        # print(arr[1])
        # exit()
        lines[id][1] = arr[1]
      # print(str(id) + ',' + pairs[0] + '\t' + pairs[1])

fout = codecs.open(data_path + "/sort_results", 'w', 'utf8')
for id, pairs in enumerate(lines):
  fout.write(str(id) + ',' + pairs[0] + '\t' + pairs[1] + '\n')
  # print()
fout.close()
