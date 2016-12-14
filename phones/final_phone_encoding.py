import codecs

with codecs.open("final_phones.txt", "r", "utf-8") as ins:
  t1 = ins.readlines()
final_output = ""
code = 0
for l in t1:
  ipa = l.split('\t')[0]
  tamil = l.split('\t')[1]
  malayalam = l.split('\t')[2][:-1]
  if ipa == "IPA":
  	continue
  for m in malayalam.split(' '):
  	if m == "":
  		continue
  	final_output += m + "\t" + str(code) + "\n"
  for t in tamil.split(' '):
  	if t == "":
  		continue
  	final_output += t + "\t" + str(code) + "\n"
  code+=1

with codecs.open("final_encoding.txt", "w", "utf-8") as f:
  f.write(final_output)