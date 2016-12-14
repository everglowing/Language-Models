import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str,
                    help='filename of text to evaluate on')
parser.add_argument('--table', type=str,
                    help='filename of IPA table', default='final_ml_phones.txt')
args = parser.parse_args()

with codecs.open(args.text, "r", encoding='utf-8') as f:
    data = f.read()

with codecs.open(args.table, "r", encoding='utf-8') as f:
    table = f.readlines()

malayalam = {}

for l in table:
	key = l.split('\t')[0]
	value = l.split('\t')[1][:-1]
	malayalam[key] = value

ml_inv = {}
for k, v in malayalam.iteritems():
    ml_inv[v] = ml_inv.get(v, [])
    ml_inv[v].append(k)

extra = []

i=0
data2 = []
while i < len(data):
	c = data[i]
	c2 = data[i:i+2]
	if c2 in malayalam and c2 not in data2:
		data2.append(malayalam)
		i+=2
	elif c2 in malayalam:
		i+=2
	else:
		i+=1
print data2

for c in data:
	if c not in malayalam and c not in extra:
		extra.append(c)

print extra


