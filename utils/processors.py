# List of text processor utilities
import collections
import numpy as np

def default_process(data):
    counter = collections.Counter(data)
    counter.update(('<S>', '</S>', 'UNK'))  # add tokens for start end and unk
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))
    return vocab

# Evaluation processors
def partial_brnn(text, vocab, back_steps, seq_length):
    x = [vocab[c] if c in vocab else vocab['UNK'] for c in text]
    x = [vocab['<S>']] + x + [vocab['</S>']]
    total_len = len(x) - 1
    # pad x so the sequence length divides it
    while len(x) % seq_length != 1:
        x.append(vocab[' '])
    y = np.array(x[1:]).reshape((-1, seq_length))
    x = np.array(x[:-1]).reshape((-1, seq_length))

    prep_array = vocab[' ']*np.ones([back_steps - 1])
    x1 = []
    for i in range(0, len(x)):
        x1.append(np.concatenate((prep_array, x[i])))
        prep_array = x[i][-1*(back_steps - 1):]
    x = np.array(x1)
    return x, y


