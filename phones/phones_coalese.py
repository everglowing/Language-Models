import codecs

with codecs.open("final_ml_phones.txt", "r", "utf-8") as ins:
  t1 = ins.readlines()
with codecs.open("final_ta_phones.txt", "r", "utf-8") as ins:
  t2 = ins.readlines()

malayalam = {}

for l in t1:
  key = l.split('\t')[0]
  value = l.split('\t')[1][:-1]
  if key in malayalam:
    print value
  malayalam[key] = value

ml_inv = {}
for k, v in malayalam.iteritems():
    ml_inv[v] = ml_inv.get(v, [])
    ml_inv[v].append(k)

tamil = {}

for l in t2:
  key = l.split('\t')[0]
  value = l.split('\t')[1][:-1]
  if key in tamil:
    print value
  tamil[key] = value

ta_inv = {}
for k, v in tamil.iteritems():
    ta_inv[v] = ta_inv.get(v, [])
    ta_inv[v].append(k)

final_output = "IPA\tTamil\tMalayalam\n"

for key, value in ml_inv.iteritems():
	if key in ta_inv:
		final_output += key + "\t"
		final_output += ' '.join(ta_inv[key]) + "\t"
		final_output += ' '.join(value) + "\n"

for key, value in ml_inv.iteritems():
	if key not in ta_inv:
		final_output += key + "\t\t"
		final_output += ' '.join(value) + "\n"

for key, value in ta_inv.iteritems():
	if key not in ml_inv:
		final_output += key + "\t"
		final_output += ' '.join(value) + "\t\n"

with codecs.open("final_phones.txt", "w", "utf-8") as f:
  f.write(final_output)


