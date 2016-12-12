import codecs

with codecs.open("tamil_phones.txt", "r", "utf-8") as ins:
  t1 = ins.readlines()
with codecs.open("tamil_wiki.txt", "r", "utf-8") as ins:
  t2 = ins.readlines()

isl_data = {}

for l in t1:
  key = l.split('\t')[0]
  value = l.split('\t')[1][:-1]
  if key in isl_data:
    print value
  isl_data[key] = value

wiki_data = {}

for l in t2:
  key = l.split(' ')[0]
  value = l.split(' ')[1][:-1]
  wiki_data[key] = value

final_list = ""

for key,value in wiki_data.iteritems():
  if key in isl_data and value != isl_data[key]:
    print key + " " + value + " " + isl_data[key]

for key,value in wiki_data.iteritems():
  final_list += key + "\t" + value + "\n"

for key,value in isl_data.iteritems():
  if key not in wiki_data:
    final_list += key + "\t" + value + "\n"

with codecs.open("final_ta_phones.txt", "w", "utf-8") as f:
  f.write(final_list)