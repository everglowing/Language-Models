# List of text processor utilities
import collections

def default_process(data):
    counter = collections.Counter(data)
    counter.update(('<S>', '</S>', 'UNK'))  # add tokens for start end and unk
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))
    return vocab

