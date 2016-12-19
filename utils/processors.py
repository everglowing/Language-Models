# List of text processor utilities
from .strings import FILES

import codecs
import collections
import numpy as np

# Helper function
def generate_ipa(ipa_file=FILES[6]):
    with codecs.open(ipa_file, "r", 'utf-8') as f:
        data = f.readlines()
    ipa_vocab = {}
    vocab_size = 0
    for l in data:
        encoding = int(l.split('\t')[1][:-1])
        ipa_vocab[l.split('\t')[0]] = encoding
        if encoding > vocab_size:
            vocab_size = encoding
    ipa_vocab['.'] = vocab_size + 1
    ipa_vocab[' '] = vocab_size + 2
    ipa_vocab['<S>'] = vocab_size + 3
    ipa_vocab['</S>'] = vocab_size + 4
    ipa_vocab['\n'] = vocab_size + 5
    ipa_vocab['UNK'] = vocab_size + 6
    vocab_size += 7
    return ipa_vocab

def rearrange_chars(chars):
    regex = ur"[\u0D00-\u0D7F\u002E]"
    ml_chars = []
    ta_chars = []
    for char in chars:
        if re.search(regex, char) is not None:
            ml_chars.append(char)
        else:
            if char == '\n' or char == ' ' or char == '<S>' or char == 'UNK' or \
               char == '</S>':
                ml_chars.append(char)
            else:
                ta_chars.append(char)
    chars_split = len(ml_chars)
    chars = ml_chars + ta_chars
    return chars, chars_split

# Text processors
def default_process(data):
    counter = collections.Counter(data)
    counter.update(('<S>', '</S>', 'UNK'))  # add tokens for start end and unk
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))
    return vocab, {}, [len(vocab)]

def ipa_process(data):
    counter = collections.Counter(data)
    counter.update(('<S>', '</S>', 'UNK'))  # add tokens for start end and unk
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))
    ipa_vocab = generate_ipa()
    return vocab, ipa_vocab, [len(vocab)]

def ipa_process_ordered(data):
    counter = collections.Counter(data)
    counter.update(('<S>', '</S>', 'UNK'))  # add tokens for start end and unk
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    chars, chars_split  = rearrange_chars(chars)
    vocab = dict(zip(chars, range(len(chars))))
    ipa_vocab = generate_ipa()
    return vocab, ipa_vocab, [chars_split, len(vocab)]

# Evaluation processors
def partial_brnn(text, vocab, ipa_vocab, seq_length, extra_data=None):
    back_steps = extra_data["back_steps"]
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
    return x, y, total_len

def brnn_gap(text, vocab, ipa_vocab, seq_length, extra_data=None):
    x = [vocab[c] if c in vocab else vocab['UNK'] for c in text]
    x = [vocab['<S>']] + x + [vocab['</S>']]
    total_len = len(x) - 1
    # pad x so the batch_size divides it
    while len(x) % seq_length != 1:
        x.append(vocab[' '])
    y = np.array(x[1:]).reshape((-1, seq_length))
    x = np.array(x[:-1]).reshape((-1, seq_length))

    app_array = vocab[' ']*np.ones([2])
    x1 = []
    for i in range(len(x)-1, 0, -1):
        x1.insert(0, np.concatenate((x[i], app_array)))
        app_array = x[i][:2]
    x = np.array(x1)
    return x, y, total_len

def phones_rnn(text, vocab, ipa_vocab, seq_length, extra_data=None):
    x = [vocab[c] if c in vocab else vocab['UNK'] for c in text]
    x = [vocab['<S>']] + x + [vocab['</S>']]
    ipa_x = [ipa_vocab[c] if c in ipa_vocab else ipa_vocab['UNK'] for c in text]
    ipa_x = [ipa_vocab['<S>']] + ipa_x + [ipa_vocab['</S>']]
    total_len = len(x) - 1
    # pad x so the batch_size divides it
    while len(x) % 200 != 1:
        x.append(vocab[' '])
        ipa_x.append(ipa_vocab[' '])
    y = np.array(x[:-1]).reshape((-1, seq_length))
    ipa_x = np.array(ipa_x[:-1]).reshape((-1, seq_length))
    return ipa_x, y, total_len

def phone_to_phone(text, vocab, ipa_vocab, seq_length, extra_data=None):
    ipa_x = [ipa_vocab[c] if c in ipa_vocab else ipa_vocab['UNK'] for c in text]
    ipa_x = [ipa_vocab['<S>']] + ipa_x + [ipa_vocab['</S>']]
    total_len = len(ipa_x) - 1
    # pad ipa_x so the batch_size divides it
    while len(ipa_x) % 200 != 1:
        ipa_x.append(ipa_vocab[' '])
    ipa_y = np.array(ipa_x[1:]).reshape((-1, seq_length))
    ipa_x = np.array(ipa_x[:-1]).reshape((-1, seq_length))
    return ipa_x, ipa_y, total_len

def phone_to_char(text, vocab, ipa_vocab, seq_length, extra_data=None):
    ipa_x = [ipa_vocab[c] if c in ipa_vocab else ipa_vocab['UNK'] for c in text]
    ipa_x = [ipa_vocab['<S>']] + ipa_x + [ipa_vocab['</S>']]
    total_len = len(ipa_x) - 1
    # pad ipa_x so the batch_size divides it
    while len(ipa_x) % 200 != 1:
        ipa_x.append(ipa_vocab[' '])
    ipa_y = np.array(ipa_x[1:]).reshape((-1, seq_length))
    ipa_x = np.array(ipa_x[:-1]).reshape((-1, seq_length))
    return ipa_x, ipa_y, total_len

def char_to_char(text, vocab, ipa_vocab, seq_length, extra_data=None):
    x = [vocab[c] if c in vocab else vocab['UNK'] for c in text]
    x = [vocab['<S>']] + x + [vocab['</S>']]
    total_len = len(x) - 1
    # pad ipa_x so the batch_size divides it
    while len(x) % 200 != 1:
        x.append(ipa_vocab[' '])
    y = np.array(x[1:]).reshape((-1, seq_length))
    x = np.array(x[:-1]).reshape((-1, seq_length))
    return x, y, total_len
